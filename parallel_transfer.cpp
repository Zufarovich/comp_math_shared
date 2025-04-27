#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <sstream>

#define _USE_MATH_DEFINES

typedef struct{
    double a;
    std::vector<std::vector<double> >* f;
    double tau;
    double h;
} Params;

int Num_x_points = 1000;
int N_x = Num_x_points;
int M_tau = 10000;

void parallel_cross_next_layer(std::vector<std::vector<double> >& grid, int num_t_layer, Params params);
void parallel_four_points_next_layer(std::vector<double>& previous, std::vector<double>& next, Params params);
void four_points_next_layer(std::vector<double>& previous, std::vector<double>& next, Params params);
void cross_next_layer(std::vector<std::vector<double> >& grid, int num_t_layer, Params params);
double phi_func(int n);

int main(int argc, char* argv[])
{
    if (argc == 1){
        printf("You have to eneter coeficient a too!\n");
        return 0;
    }
    double a = atof(argv[1]);
    int commsize, my_rank, value;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int last = commsize - 1;
    int remainder = Num_x_points % commsize;
    int product = Num_x_points / commsize;

    std::vector<std::vector<double>> f;

    std::vector<std::vector<double> > grid_solution;

    int stop_id, start_id;

    if (my_rank < remainder){
        if(!my_rank)
            N_x = product + 2;
        else
            N_x = product + 3;

        f.resize(M_tau, std::vector<double>(N_x));
        grid_solution.resize(M_tau, std::vector<double>(N_x));
        start_id = product * my_rank + my_rank;
        stop_id = start_id + product;

    }
    else{
        if(!my_rank || my_rank == last)
            N_x = product + 1;
        else
            N_x = product + 2;

        f.resize(M_tau, std::vector<double>(N_x));
        grid_solution.resize(M_tau, std::vector<double>(N_x));
        start_id = product * my_rank + remainder;
        stop_id = start_id + product - 1;
    }

    double tau = 1 / ((double) M_tau);
    double h = 1 / ((double) Num_x_points);

    if(a * tau / h > 1){
        printf("The Courant Number %g > 1!\n", a * tau / h);
        abort();
    }

    for (auto& row : f) {
        std::fill(row.begin(), row.end(), 0.0);
    }

    for(int n = start_id; n < stop_id + int(my_rank != last); n++){
        grid_solution[0][n - start_id + int(my_rank != 0)] = phi_func(n);
    }

    for(int k = 0; k < M_tau & !my_rank; k++){
        grid_solution[k][0] = 0;
    }

    Params pampam = {a, &f, tau, h};
    double coef = tau * a /  h;
    if(!my_rank){
        MPI_Send(&grid_solution[0][N_x - 2], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&grid_solution[0][N_x - 1], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
    }else if(my_rank == last){
        MPI_Recv(&grid_solution[0][0], 1, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&grid_solution[0][1], 1, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);     
    }else{
        MPI_Recv(&grid_solution[0][0], 1, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&grid_solution[0][1], 1, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
        
        MPI_Send(&grid_solution[0][N_x - 2], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&grid_solution[0][N_x - 1], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
    }

    parallel_four_points_next_layer(grid_solution[0], grid_solution[1], pampam);
    if(my_rank == last)
        grid_solution[1][N_x - 1]  = grid_solution[0][N_x - 1] - 
        coef * (grid_solution[0][N_x - 1] - grid_solution[0][N_x - 2])+
        tau * f[0][N_x - 1];

    for(int j = 2; j < M_tau; j++){
        if(!my_rank){
            MPI_Send(&grid_solution[j - 1][N_x - 2], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&grid_solution[j - 1][N_x - 1], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
        }else if(my_rank == last){
            MPI_Recv(&grid_solution[j - 1][0], 1, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&grid_solution[j - 1][1], 1, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);     
        }else{
            MPI_Recv(&grid_solution[j - 1][0], 1, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&grid_solution[j - 1][1], 1, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
            
            MPI_Send(&grid_solution[j - 1][N_x - 2], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&grid_solution[j - 1][N_x - 1], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
        }

        parallel_cross_next_layer(grid_solution, j, pampam);

        if(my_rank == last)
            grid_solution[j][N_x - 1] = grid_solution[j - 1][N_x - 1] - 
                                        coef * (grid_solution[j - 1][N_x - 1] - grid_solution[j - 1][N_x - 2])+
                                        tau * f[j - 1][N_x - 1];
    }

    std::stringstream filename;
    filename << "results/results_" << my_rank << ".txt";
    std::string name = filename.str();

    FILE* save = fopen(name.c_str(), "w");
    if (save == nullptr) {
        printf("Ошибка открытия файла!\n");
        return 1;
    }

    fprintf(save, "%d %d\n", my_rank, commsize);
    for (int k = 0; k < M_tau; ++k) {
        for (int n = 0; n < N_x; ++n) {
            fprintf(save, "%g ", grid_solution[k][n]);
        }
        fprintf(save, "\n");
    }


    MPI_Finalize();
    return 0;
}

void four_points_next_layer(std::vector<double>& previous, std::vector<double>& next, Params params) {
    double coef = params.tau * params.a / (2.0 * params.h);

    for (int i = 1; i < N_x - 1; i++) {
        next[i] = previous[i] - coef * (previous[i + 1] - previous[i - 1]);
    }

    next[N_x - 1] = previous[N_x - 1] - 2 * coef * (previous[N_x - 1] - previous[N_x - 2]);
}

void cross_next_layer(std::vector<std::vector<double> >& grid, int num_t_layer, Params params){
    double coef = params.tau * params.a /  params.h;

    for (int i = 1; i < N_x - 1; i ++){
        grid[num_t_layer][i] = grid[num_t_layer - 2][i] + 2 * params.tau * (*params.f)[num_t_layer - 1][i] -
            coef * (grid[num_t_layer - 1][i + 1] - grid[num_t_layer - 1][i - 1]);
    }

    grid[num_t_layer][N_x - 1] = grid[num_t_layer - 1][N_x-1] - coef *
    (grid[num_t_layer - 1][N_x-1] - grid[num_t_layer - 1][N_x-2]);
}

void parallel_four_points_next_layer(std::vector<double>& previous, std::vector<double>& next, Params params) {
    double coef = params.tau * params.a / (2.0 * params.h);

    for (int i = 1; i < N_x - 1; i++) {
        next[i] = previous[i] - coef * (previous[i + 1] - previous[i - 1]) + params.tau * (*params.f)[1][i];
    }
}

void parallel_cross_next_layer(std::vector<std::vector<double> >& grid, int num_t_layer, Params params){
    double coef = params.tau * params.a /  params.h;

    for (int i = 1; i < N_x - 1; i ++){
        grid[num_t_layer][i] = grid[num_t_layer - 2][i] + 2 * params.tau * (*params.f)[num_t_layer - 1][i] -
            coef * (grid[num_t_layer - 1][i + 1] - grid[num_t_layer - 1][i - 1]);
    }
}

double phi_func(int n){
    double x = (double)n / (double)Num_x_points;
    if(x < 0.5)
    {
        float sin_val = sin(2 * M_PI * x);
        return  sin_val * sin_val;
    }
    else
        return 0;
}
