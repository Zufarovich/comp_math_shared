#include <iostream>
#include <vector>
#include <cmath>
#include <thread>         // Use std::thread
#include <mutex>          // Use std::mutex
#include <condition_variable> // Use std::condition_variable
#include <unistd.h>       // For sysconf
#include <stdexcept>      // For runtime_error
#include <iomanip>        // For std::setprecision
#include <atomic>         // For atomic counters if needed

// --- Configuration ---
const double INTEGRATION_EPSILON = 1e-7;
const int LOCAL_STACK_THRESHOLD = 10;
const int MAX_GLOBAL_TASKS = 10000;

// --- Function to Integrate ---
double fun(double x) {
    if (x == 0.0) {
        // Return NaN for problematic points, let caller handle
        return NAN; // Or std::numeric_limits<double>::quiet_NaN();
    }
    // return 1.0 / x;
    double sin2 = sin(1.0 / x);
     return 1/(x*x)*sin2*sin2;
    // return 4.0 / (1.0 + x * x);
}

// --- Break Condition ---
bool BreakCond(double sacb, double sab) {
    // Use relative and absolute tolerance
    return std::abs(sacb - sab) > INTEGRATION_EPSILON * (1.0 + std::abs(sab));
}

// --- Task Structure ---
struct Task {
    double a, b;
    double fa, fb;
    double sab;

    // Explicit constructors
    Task() : a(0), b(0), fa(0), fb(0), sab(0) {}
    Task(double _a, double _b, double _fa, double _fb, double _sab)
        : a(_a), b(_b), fa(_fa), fb(_fb), sab(_sab) {}
};

// --- Global Shared Data ---
struct SharedData {
    std::vector<Task> list_of_tasks;
    int ntask; // How many tasks are currently IN the list
    std::atomic<int> nactive; // How many threads are actively processing
    double s_all;
    int nproc;
    int maxtask;
    bool terminate_signal; // Flag to signal termination

    std::mutex list_mutex;       // Protects list_of_tasks and ntask
    std::mutex sum_mutex;        // Protects s_all
    std::condition_variable task_available_cv; // Signal for new tasks

    SharedData(int max_tasks = MAX_GLOBAL_TASKS)
        : ntask(0), nactive(0), s_all(0.0), nproc(0), maxtask(max_tasks), terminate_signal(false) {
        list_of_tasks.resize(maxtask);
    }
} sdat; // Global instance

// --- Helper Macros/Functions for Stack Operations ---
// Global Stack Access (MUST be called while holding list_mutex)
void PUT_TO_GLOBAL_STACK_LOCKED(double a, double b, double fa, double fb, double sab) {
    if (sdat.ntask >= sdat.maxtask) {
        // This shouldn't happen if MAX_GLOBAL_TASKS is large enough
        // and checks are done correctly, but good to have a guard.
        std::cerr << "Warning: Global stack full during PUT attempt." << std::endl;
        return; // Or throw, depending on desired behavior
    }
    sdat.list_of_tasks[sdat.ntask++] = Task(a, b, fa, fb, sab);
}
Task GET_OF_GLOBAL_STACK_LOCKED() {
    // Assumes ntask > 0 check was done before calling
    return sdat.list_of_tasks[--sdat.ntask];
}

// Local Stack (specific to each thread)
struct LocalStack {
    std::vector<Task> stack;
    int sp; // Points to the next free slot (stack size)

    LocalStack() : sp(0) {
        stack.resize(100); // Initial reasonable size
    }

    void PUT_TO_LOCAL_STACK(double a, double b, double fa, double fb, double sab) {
        if (sp >= stack.size()) {
            stack.resize(stack.size() * 2); // Grow if needed
        }
        stack[sp++] = Task(a, b, fa, fb, sab);
    }

    Task GET_FROM_LOCAL_STACK() {
        if (sp <= 0) throw std::runtime_error("Local stack underflow");
        return stack[--sp];
    }

    int get_sp() const {
        return sp;
    }
};

// --- Worker Thread Function ---
void worker_thread_func() {
    LocalStack local_stack;
    double thread_local_sum = 0.0;
    bool thread_active = false; // Was this thread processing a real task?

    while (true) {
        Task current_task;
        bool task_taken = false;

        // --- Wait for and get a task from the global queue ---
        { // Lock scope for list_mutex and condition variable
            std::unique_lock<std::mutex> lock(sdat.list_mutex);

            // Wait condition: Wait if NO tasks AND termination not signaled
            sdat.task_available_cv.wait(lock, [] {
                return sdat.ntask > 0 || sdat.terminate_signal;
            });

            // If termination signaled AND no tasks left, exit loop
            if (sdat.terminate_signal && sdat.ntask == 0) {
                break; // Exit the main while loop
            }

            // If woken up, there must be a task (or termination)
            if (sdat.ntask > 0) {
                current_task = GET_OF_GLOBAL_STACK_LOCKED(); // Also decrements ntask
                task_taken = true;
            } else {
                // This case (woken up by terminate_signal but ntask is 0)
                // should be handled by the break above. If we reach here,
                // it might be a spurious wakeup without tasks, just continue waiting.
                continue;
            }
        } // Release lock (unique_lock destructor)

        // --- Process the obtained task ---
        if (task_taken) {
             // Check for the special terminal task (a > b)
            bool is_terminal = (current_task.a > current_task.b); // Use the original signal
            if (is_terminal) {
                 // We received the signal to terminate
                 sdat.terminate_signal = true; // Ensure flag is set
                 sdat.task_available_cv.notify_all(); // Wake others to check termination
                 break; // Exit loop
            }


            // If it's a real task, mark thread as active
            sdat.nactive++; // Atomic increment
            thread_active = true;
            local_stack.sp = 0; // Reset local stack for this new global task

            double a = current_task.a;
            double b = current_task.b;
            double fa = current_task.fa;
            double fb = current_task.fb;
            double sab = current_task.sab;

            // Inner loop for adaptive refinement
            while (true) {
                double c = (a + b) / 2.0;
                double fc = fun(c);

                if (std::isnan(fc)) {
                    // Problem evaluating function. Add current best estimate and stop refining this path.
                    std::cerr << "Warning: fun(c) returned NaN for c = " << c << ". Adding sab = " << sab << std::endl;
                    thread_local_sum += sab; // Add the estimate for the interval [a, b]
                    // Try to get work from local stack
                    if (local_stack.get_sp() == 0) break; // Inner loop break
                    Task next = local_stack.GET_FROM_LOCAL_STACK();
                    a = next.a; b = next.b; fa = next.fa; fb = next.fb; sab = next.sab;
                    continue; // Process the popped task
                }

                double sac = (fa + fc) * (c - a) / 2.0;
                double scb = (fc + fb) * (b - c) / 2.0;
                double sacb = sac + scb;

                if (!BreakCond(sacb, sab)) {
                    thread_local_sum += sacb; // Precision met, add result
                    if (local_stack.get_sp() == 0) {
                        break; // Inner loop break: Current global task fully processed
                    } else {
                        // Get next sub-task from local stack
                        Task next = local_stack.GET_FROM_LOCAL_STACK();
                        a = next.a; b = next.b; fa = next.fa; fb = next.fb; sab = next.sab;
                        // Continue inner loop with the popped task
                    }
                } else {
                    // Precision not met, push right interval, continue with left
                    local_stack.PUT_TO_LOCAL_STACK(c, b, fc, fb, scb);
                    b = c;
                    fb = fc;
                    sab = sac;
                    // Continue inner loop refining the new [a, b] (which is the old [a, c])
                }

                // --- Check for Load Balancing (Donating Tasks) ---
                if (local_stack.get_sp() > LOCAL_STACK_THRESHOLD) {
                     bool potentially_idle = false;
                     { // Check global state briefly
                         std::lock_guard<std::mutex> lock(sdat.list_mutex);
                         potentially_idle = (sdat.ntask == 0);
                     } // Release lock

                     if (potentially_idle) { // Donate only if global queue might be empty
                        std::lock_guard<std::mutex> lock(sdat.list_mutex);
                        // Re-check ntask while holding lock
                        if (sdat.ntask == 0) {
                            int tasks_moved = 0;
                            while (local_stack.get_sp() > 1 && sdat.ntask < sdat.maxtask) {
                                Task task_to_move = local_stack.GET_FROM_LOCAL_STACK();
                                PUT_TO_GLOBAL_STACK_LOCKED(task_to_move.a, task_to_move.b, task_to_move.fa, task_to_move.fb, task_to_move.sab);
                                tasks_moved++;
                            }
                            if (tasks_moved > 0) {
                                // Notify potentially waiting threads *after* adding tasks and releasing lock
                                lock.~lock_guard(); // Explicitly release before notifying
                                sdat.task_available_cv.notify_all(); // Wake up others
                            }
                        }
                     }
                } // End load balancing check
            } // End inner while loop (adaptive refinement)

            // Task processing finished, decrement active count
            sdat.nactive--; // Atomic decrement
            thread_active = false;

            // --- Check if this is the last active thread ---
            if (sdat.nactive == 0) {
                 std::lock_guard<std::mutex> lock(sdat.list_mutex);
                 // Re-check nactive and ntask together under lock
                 if (sdat.nactive == 0 && sdat.ntask == 0 && !sdat.terminate_signal) {
                     // This looks like the end of computation
                     std::cout << "Thread " << std::this_thread::get_id() << " appears to be last active, signaling termination." << std::endl;

                     // Add terminal tasks for all threads
                     int terminal_added = 0;
                     for (int i = 0; i < sdat.nproc; ++i) {
                         if (sdat.ntask < sdat.maxtask) {
                             PUT_TO_GLOBAL_STACK_LOCKED(2.0, 1.0, 0.0, 0.0, 0.0); // Terminal task (a > b)
                             terminal_added++;
                         } else {
                             std::cerr << "Warning: Global stack full, cannot add terminal task for thread " << i << std::endl;
                             // If we can't add terminal tasks, termination might fail!
                             // Setting the flag is crucial here.
                         }
                     }
                     sdat.terminate_signal = true; // Set the termination flag

                     if (terminal_added > 0 || sdat.terminate_signal) {
                        lock.~lock_guard(); // Release lock before notify
                        sdat.task_available_cv.notify_all(); // Wake up all threads to check termination
                     }
                 }
            } // End last active check

        } // End if(task_taken)

    } // End main while loop

    // --- Add local sum to global sum ---
    if (thread_local_sum != 0.0) {
        std::lock_guard<std::mutex> lock(sdat.sum_mutex);
        sdat.s_all += thread_local_sum;
    }
    // std::cout << "Thread " << std::this_thread::get_id() << " finished." << std::endl;
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    double A = 1e-5;
    double B = 1.0; // For 4/(1+x*x), A=0, B=1 gives PI
    int p = 0;

    if (argc < 2) {
        p = std::thread::hardware_concurrency(); // Get hardware threads
        if (p == 0) p = 2; // Default if detection fails
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

    // Initialize SharedData (constructor handles mutexes etc.)
    sdat.nproc = p;
    sdat.maxtask = MAX_GLOBAL_TASKS; // Set max size
    sdat.list_of_tasks.resize(sdat.maxtask); // Ensure vector has space

    // --- Check initial function values ---
    double fA = fun(A);
    double fB = fun(B);
    if (std::isnan(fA) || std::isnan(fB)) {
        std::cerr << "Error: Function returns NaN at interval endpoints (" << A << ", " << B << "). Cannot integrate." << std::endl;
        return 1;
    }
    double sAB = (fA + fB) * (B - A) / 2.0; // Initial trapezoid estimate

    // --- Add initial task ---
    { // Lock scope
        std::lock_guard<std::mutex> lock(sdat.list_mutex);
        PUT_TO_GLOBAL_STACK_LOCKED(A, B, fA, fB, sAB);
    } // Release lock
    sdat.task_available_cv.notify_one(); // Signal one thread that a task is ready

    // --- Create and start worker threads ---
    std::vector<std::thread> worker_threads;
    std::cout << "Starting " << p << " worker threads..." << std::endl;
    for (int i = 0; i < p; ++i) {
        try {
            worker_threads.emplace_back(worker_thread_func);
        } catch (const std::system_error& e) {
            std::cerr << "Failed to create thread " << i << ": " << e.what() << std::endl;
            // Attempt to signal existing threads to terminate early might be needed here
            sdat.terminate_signal = true;
            sdat.task_available_cv.notify_all();
            // Join threads that were created
            for(int j = 0; j < i; ++j) {
                if(worker_threads[j].joinable()) worker_threads[j].join();
            }
            return 1;
        }
    }

    // --- Wait for all threads to finish ---
    std::cout << "Waiting for threads to finish..." << std::endl;
    for (auto& th : worker_threads) {
        if (th.joinable()) {
            th.join();
        }
    }
    std::cout << "All threads finished." << std::endl;

    // --- Print result ---
    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Integration Result (s_all): " << sdat.s_all << std::endl;

     // Example analytical results for comparison:
     // Integral of sin(1/x) from 1e-5 to 1: Approx 0.504066 (using WolframAlpha)
     // Integral of 4/(1+x^2) from 0 to 1: PI = 3.1415926535...

    return 0;
}