#include <iostream>
#include <vector>
#include <cmath>
#include <thread> 
#include <mutex>          
#include <condition_variable> 
#include <unistd.h>      
#include <stdexcept>     
#include <iomanip>       
#include <atomic>   
#include <time.h>

const double INTEGRATION_EPSILON = 1e-5; //Задаем точность вычисления
const int LOCAL_STACK_THRESHOLD = 10;    //Максимальное число заданий локальном стэке
const int MAX_GLOBAL_TASKS = 10000;      //Максимальное число заданий в глобальном стэке

double fun(double x) {
    if (x == 10.0) {
        // Возврацает NAN в точках, где функция не определена
        return NAN; 
    }
    // return 1.0 / x;
    //double sin2 = sin(1.0 / x);
     //return 1/(x*x)*sin2*sin2;
    double cos_s = cos(1 / (10 - x));
     return cos_s;
    //return 4.0 / (1.0 + x * x);
}

// Break Condition - условие достижения требуемой точности 
// Если точность достигнуте, возвращаем false
bool BreakCond(double sacb, double sab) {
    // Используем и абсолютную погрешность и относительную
    return std::abs(sacb - sab) > INTEGRATION_EPSILON * (1.0 + std::abs(sab));
}

// Структура задачи
struct Task {
    double a, b;
    double fa, fb;
    double sab;

    // Добавим два конструктора
    Task() : a(0), b(0), fa(0), fb(0), sab(0) {}
    Task(double _a, double _b, double _fa, double _fb, double _sab)
        : a(_a), b(_b), fa(_fa), fb(_fb), sab(_sab) {}
};

// Global Shared Data - общая конфигурация, что-то вроде коммуникатора
struct SharedData {
    std::vector<Task> list_of_tasks; // Сам стэк задач
    int ntask; // Сколько на данный момент задач в стэке 
    std::atomic<int> nactive; // Сколько на данный момент нитей выполняютс задачу  
    double s_all; // Переменная для подстчета интеграла
    int nproc; // Количество потоков-работников
    int maxtask; // Максимальное число задач в стэке
    bool terminate_signal; // Флаг для терминальной инструкции

    std::mutex list_mutex;       // Mьютекс для доступа к глобальному стэку 
    std::mutex sum_mutex;        // Мьютекс для доступа к переменной подсчета интеграла
    std::condition_variable task_available_cv; // Условная переменная для пробуждения потоков

    // Добавим конструктор
    SharedData(int max_tasks = MAX_GLOBAL_TASKS)
        : ntask(0), nactive(0), s_all(0.0), nproc(0), maxtask(max_tasks), terminate_signal(false) {
        list_of_tasks.resize(maxtask);
    }
} sdat; // Создаем структуру в единственном экземпляре под именем sdat

// Далее описываются вспомогательные функции для работы со стеками

// Добавление в глобальный стек (необходимо делать под мьютексом!!!)
void PUT_TO_GLOBAL_STACK_LOCKED(double a, double b, double fa, double fb, double sab) {
    if (sdat.ntask >= sdat.maxtask) {
        // Если число доступных задач в глобальном стеке превышено, то выдаем ошибку
        std::cerr << "Warning: Global stack full during PUT attempt." << std::endl;
        return; 
    }
    // Добавляем новую задачу и смещаем указатель
    sdat.list_of_tasks[sdat.ntask++] = Task(a, b, fa, fb, sab);
}
Task GET_OF_GLOBAL_STACK_LOCKED() {
    // Предполагаем, что доступ к записи можно будет получить
    // только в случае, если стэк не пуст
    return sdat.list_of_tasks[--sdat.ntask];
}

// Структура локального стэка
struct LocalStack {
    std::vector<Task> stack;
    int sp; // Указатель на СВОБОДНЫЙ слот

    // Конструктор по умолчанию
    LocalStack() : sp(0) {
        stack.resize(100); // Размер по умолчанию равен 100
    }

    // Добавление задачи в локальный стэк
    void PUT_TO_LOCAL_STACK(double a, double b, double fa, double fb, double sab) {
        if (sp >= stack.size()) {
            stack.resize(stack.size() * 2); // Если в стэке не хватает месте - увеличиваем вдвое
        }
        stack[sp++] = Task(a, b, fa, fb, sab);
    }
    // Получение задачи из стэка. Если в стэке пусто, выдаем ошибку
    Task GET_FROM_LOCAL_STACK() {
        if (sp <= 0) throw std::runtime_error("Local stack underflow");
        return stack[--sp];
    }
    // Получить текущее значение указателя на СВОБОДНЫЙ слот для задачи
    int get_sp() const {
        return sp;
    }
};

// Функция для потока_работника
void worker_thread_func() {
    LocalStack local_stack; // Инициализация локального стэка
    double thread_local_sum = 0.0; // Локальная сумма задачи
    bool thread_active = false; // Выполняет ли поток задачу в данный момент?

    // Основной цикл работы. Поток работает до приходе терминальной задачи.
    while (true) {
        Task current_task; // Инициаллизируем текущую задачу
        bool task_taken = false; // Фалаг: получена ли задача?

        // Получение задачи из глобального стэка
        { // Входим в мьютекс для доступа к глобальному стэку
            std::unique_lock<std::mutex> lock(sdat.list_mutex);

            // Если не терминальная команда и глобальный стэк задач пуст
            // поток засыпает под условной переменной
            sdat.task_available_cv.wait(lock, [] {
                return sdat.ntask > 0 || sdat.terminate_signal;
            });

            // Если стек пуст и пришла терминальная команда,
            // то выходим из цикла
            if (sdat.terminate_signal && sdat.ntask == 0) {
                break; 
            }

            // Если процесс добрался до этого момента, то или в глобальном стэке есть задачи
            // или пришла терминальная команда
            if (sdat.ntask > 0) {
                current_task = GET_OF_GLOBAL_STACK_LOCKED(); // Поток под мьютексом берет задачу
                                                             // из глобального стека
                task_taken = true; // Поток выставляет флаг о взятой задаче
            } else {
                // Если в стэке нет задач, но при этом пришел терминальный 
                // сигнал, то произошло ложное пробуждение, поэтому 
                // поток должен продолжить спать
                continue;
            }
        } // В этот момент поток возвращает мьютекс

        // Обработка полученного из глобального стэка задания
        if (task_taken) {
             // Проверка специально терминального задания (a > b)
            bool is_terminal = (current_task.a > current_task.b); 
            if (is_terminal) {
                 // Поток получил специальную задачу на прекращение
                 sdat.terminate_signal = true; // Убедимся в том, что терминальный сигнал включен
                 sdat.task_available_cv.notify_all(); // Поток будит остальные, чтобы те проверили
                                                      // терминальный сигнал
                 break; // Выходим из цикла
            }


            // При достижении этого участка функии поток убежден в том, что
            // он получил настоящую задачу
            sdat.nactive++; // Атомиком инкрементируем переменную рабртающиц потоков
            thread_active = true;
            local_stack.sp = 0; // Сбрасываем значение локального стэка

            double a = current_task.a;
            double b = current_task.b;
            double fa = current_task.fa;
            double fb = current_task.fb;
            double sab = current_task.sab;

            // Цикл для обработки полученной задачи
            while (true) {
                double c = (a + b) / 2.0;
                double fc = fun(c);

                if (std::isnan(fc)) {
                    // Обрабатываем случай, когда функция не определена в точке C
                    std::cerr << "Warning: fun(c) returned NaN for c = " << c << ". Adding sab = " << sab << std::endl;
                    thread_local_sum += sab; // Прибавляем к локльной сумме оценку для интеграла по трапеции

                    // Работаем с локальным стэком
                    if (local_stack.get_sp() == 0) break; // Если стэк пустой, то выходим
                    Task next = local_stack.GET_FROM_LOCAL_STACK(); // Иначе, берем задачу
                    a = next.a; b = next.b; fa = next.fa; fb = next.fb; sab = next.sab;
                    continue; // Поток уходит в начало цикла, чтобы начать обрабатывать
                              // новую задачу.
                }

                double sac = (fa + fc) * (c - a) / 2.0;
                double scb = (fc + fb) * (b - c) / 2.0;
                double sacb = sac + scb;

                // Случай, когда точность достигнута
                if (!BreakCond(sacb, sab)) {
                    thread_local_sum += sacb; // Добавляем в локальную сумму
                    if (local_stack.get_sp() == 0) {
                        break; // Текущая задача из глобально стэка выполнена, выходим из цикла
                    } else {
                        // Иначе берем новую задачу из локального стека
                        Task next = local_stack.GET_FROM_LOCAL_STACK();
                        a = next.a; b = next.b; fa = next.fa; fb = next.fb; sab = next.sab;
                        // Продолжаем цикл с новой задачей
                    }
                } else {
                    // Точность не достигнута, поток кладет правую часть задачи в стек
                    // и работает с левой
                    local_stack.PUT_TO_LOCAL_STACK(c, b, fc, fb, scb);
                    b = c;
                    fb = fc;
                    sab = sac;
                }

                // Балансировка задач
                if (local_stack.get_sp() > LOCAL_STACK_THRESHOLD) {
                     bool potentially_idle = false; // Флаг пустоты глобального стэка
                     { // Поток проверяет, не пуст ли глобатьный стэк
                         std::lock_guard<std::mutex> lock(sdat.list_mutex);
                         potentially_idle = (sdat.ntask == 0);
                     } // Выход из мьютекса

                     if (potentially_idle) { // Под мьютексом поток заходит в стэк и заполняет
                                             // своими задачами
                        std::lock_guard<std::mutex> lock(sdat.list_mutex);
                        // Перепроверим, действительно ли стэк пуст
                        if (sdat.ntask == 0) {
                            int tasks_moved = 0; // Число перемещенных задач
                            while (local_stack.get_sp() > 1 && sdat.ntask < sdat.maxtask) {
                                Task task_to_move = local_stack.GET_FROM_LOCAL_STACK();
                                PUT_TO_GLOBAL_STACK_LOCKED(task_to_move.a, task_to_move.b, task_to_move.fa, task_to_move.fb, task_to_move.sab);
                                tasks_moved++;
                            }
                            if (tasks_moved > 0) {
                                // Если поток переместил некоторое число задач в стэк
                                lock.~lock_guard(); // Он выходит из мьютекса, освобождая доступ к стэку
                                sdat.task_available_cv.notify_all(); // И будет остальные потоки, ждущие задач
                            }
                        }
                     }
                } // Завершение балансировки задач
            } // Завершение внутреннего цикла

            // Поток больше не обрабатывает задачу
            sdat.nactive--; // 
            thread_active = false;

            // Поток проверяет, не является ли он последним
            if (sdat.nactive == 0) {
                 std::lock_guard<std::mutex> lock(sdat.list_mutex);
                 // Под мьютексом проверяем, что действительно в стэке нет задач и ни один поток
                 // не работает над задачей, то есть все они спят под условной переменной
                 if (sdat.nactive == 0 && sdat.ntask == 0 && !sdat.terminate_signal) {
                     // Если поток достигает этогй строки, то он понимает, что вычисления завешрены
                     std::cout << "Thread " << std::this_thread::get_id() << " appears to be last active, signaling termination." << std::endl;

                     // Отправляем всем потокам специальную терминальную задачу
                     int terminal_added = 0;
                     for (int i = 0; i < sdat.nproc; ++i) {
                         if (sdat.ntask < sdat.maxtask) {
                             PUT_TO_GLOBAL_STACK_LOCKED(2.0, 1.0, 0.0, 0.0, 0.0); 
                             // Та самая терминальная задача (a > b)
                             terminal_added++;
                         } else {
                             std::cerr << "Warning: Global stack full, cannot add terminal task for thread " << i << std::endl;
                             // Если вдруг глобальный стек оказался переполненным, то программа
                             // сработала неправильно. Выдаем ошибку
                         }
                     }
                     sdat.terminate_signal = true; // Выставляем флаг терминальной команды

                     if (terminal_added > 0 || sdat.terminate_signal) {
                        lock.~lock_guard(); // Выходим из мьютекса доступа к глобальному стэку
                        sdat.task_available_cv.notify_all(); // Будим остальные потоки, чтобы те обработали
                        // терминальную задачу
                     }
                 }
            } // Конец участка, обрабатывающего последний поток
        } // Конец участка (task_taken)

    } // Завершение основного цикла

    // Каждый процесс в конце работы должен прибавить результат своих вычислений к 
    // общей сумме
    if (thread_local_sum != 0.0) {
        std::lock_guard<std::mutex> lock(sdat.sum_mutex);
        sdat.s_all += thread_local_sum;
    }
    std::cout << "Thread " << std::this_thread::get_id() << " finished." << std::endl;
}

// Основная функция
int main(int argc, char* argv[]) {
    double total_time = 0.0;
    struct timespec start, end;

    double A = 0;
    double B = 9.99; // Заданная функция  4/(1+x*x) на этом участке дает число пи
    int p = 0;

    if (argc < 2) {
        p = std::thread::hardware_concurrency(); // Если при запуске не укзано число потоков
        // то по умолчанию выставляется 2 потока
        if (p == 0) p = 2; 
        std::cout << "Number of threads not specified, using hardware concurrency: " << p << std::endl;
    } else {
        try {
            p = std::stoi(argv[1]);
            if (p <= 0) {
                std::cerr << "Invalid number of threads specified." << std::endl;
                return 1;
            }
            std::cout << "Using specified number of threads: " << p << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Invalid number of threads argument: " << e.what() << std::endl;
            return 1;
        }
    }

    // Инициализируем SharedData 
    sdat.nproc = p;
    sdat.maxtask = MAX_GLOBAL_TASKS; 
    sdat.list_of_tasks.resize(sdat.maxtask); 

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Проверяем начальные значения функции
    double fA = fun(A);
    double fB = fun(B);
    if (std::isnan(fA) || std::isnan(fB)) {
        std::cerr << "Error: Function returns NaN at interval endpoints (" << A << ", " << B << "). Cannot integrate." << std::endl;
        return 1;
    }
    double sAB = (fA + fB) * (B - A) / 2.0; // Исходная оценка трапецией

    // Добавляем исходное задание
    { // На всякий случай добавим обратимся к стэку по мьютексу
        std::lock_guard<std::mutex> lock(sdat.list_mutex);
        PUT_TO_GLOBAL_STACK_LOCKED(A, B, fA, fB, sAB);
    } // Возвращаем мьютекс
    //sdat.task_available_cv.notify_one(); // Будим один поток
    // Как только его стек достигнет определенного значения,
    // а это произойдет, если задача достаточно большая, то он позовет
    // на помощь остальные потоки

    // Создаем и запускаем потоки
    std::vector<std::thread> worker_threads;
    std::cout << "Starting " << p << " worker threads..." << std::endl;
    for (int i = 0; i < p; ++i) {
        try {
            worker_threads.emplace_back(worker_thread_func);
        } catch (const std::system_error& e) {
            std::cerr << "Failed to create thread " << i << ": " << e.what() << std::endl;
            // Завершаем работу потоков раньше времении в связи с ошибкой
            sdat.terminate_signal = true;
            sdat.task_available_cv.notify_all();
            // Дожидаемся завершения всех потоков
            for(int j = 0; j < i; ++j) {
                if(worker_threads[j].joinable()) worker_threads[j].join();
            }
            return 1;
        }
    }

    // Ждем завершения всех потоков
    std::cout << "Waiting for threads to finish..." << std::endl;
    for (auto& th : worker_threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    std::cout << "All threads finished." << std::endl;

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + 
                            (end.tv_nsec - start.tv_nsec) / 1e9;

    std::cout << "Elapsed time:" << elapsed_time << std::endl;
        
    // Выводим результат
    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Integration Result (s_all): " << sdat.s_all << std::endl;

     // Integral of 4/(1+x^2) from 0 to 1: PI = 3.1415926535...

    return 0;
}