#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cstdlib>

#define _USE_MATH_DEFINES

typedef struct{
    double a;
    std::vector<std::vector<double> >* f;
    double tau;
    double h;
} Params;

int N_x = 1000;
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


    int remainder = N_x % commsize;
    int product = N_x / commsize;

    std::vector<std::vector<double>> f;

    std::vector<std::vector<double> > grid_solution;

    int stop_id, start_id;

    if (my_rank < remainder){
        N_x = product + 2;
        f.resize(M_tau, std::vector<double>(N_x));
        grid_solution.resize(M_tau, std::vector<double>(N_x));
        start_id = product * my_rank + my_rank;
        stop_id = start_id + product;

    }
    else{
        N_x = product + 1;
        f.resize(M_tau, std::vector<double>(N_x));
        grid_solution.resize(M_tau, std::vector<double>(N_x));
        start_id = product * my_rank + remainder;
        stop_id = start_id + product - 1;
    }

    
    double tau = 1 / ((double) M_tau);
    double h = 1 / ((double) N_x);

    if(a * tau / h > 1){
        printf("The Courant Number %g > 1!\n", a * tau / h);
        abort();
    }

    for (auto& row : f) {
        std::fill(row.begin(), row.end(), 0.0);
    }

    for(int n = 0; n < N_x; n++){
        grid_solution[0][n] = phi_func(n);
    }
    for(int k = 0; k < M_tau; k++){
        grid_solution[k][0] = 0;
    }

    Params pampam = {a, &f, tau, h};
    double coef = tau * a / (2.0 * h);
    parallel_four_points_next_layer(grid_solution[0], grid_solution[1], pampam);
    if(my_rank == commsize - 1)
        grid_solution[1][N_x - 1]  = grid_solution[0][N_x - 1] - 2 * coef * (grid_solution[0][N_x - 1] - grid_solution[0][N_x - 2]);


    for(int j = 2; j < M_tau; j++){
        if(!my_rank){
            MPI_Send(&grid_solution[i - 2][N_x - 1], 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recieve(&, 1, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD, &status);
        }

        parallel_cross_next_layer(grid_solution, j, pampam);
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
        next[i] = previous[i] - coef * (previous[i + 1] - previous[i - 1]);
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
    double x = (double)n / (double)N_x;
    if(x < 0.5)
    {
        float sin_val = sin(2 * M_PI * x);
        return  sin_val * sin_val;
    }
    else
        return 0;
}
