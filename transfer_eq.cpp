#include <stdio.h>
//#include <mpi.h>
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


void four_points_next_layer(std::vector<double>& previous, std::vector<double>& next, Params params);
void cross_next_layer(std::vector<std::vector<double> >& grid, int num_t_layer, Params params);
double phi_func(int n);


int main(int argc, char* argv[]){
    double a = atof(argv[1]);

    double phi[N_x];
    double tau = 1 / ((double) M_tau);
    double h = 1 / ((double) N_x);
    std::vector<std::vector<double> > f(M_tau, std::vector<double>(N_x));

    if(a * tau / h > 1){
        printf("The Courant Number %g > 1!\n", a * tau / h);
        abort();
    }

    for (auto& row : f) {
        std::fill(row.begin(), row.end(), 0.0);
    }

    std::vector<std::vector<double> > grid_solution(M_tau, std::vector<double>(N_x));

    for(int n = 0; n < N_x; n++){
        grid_solution[0][n] = phi_func(n);
    }
    for(int k = 0; k < M_tau; k++){
        grid_solution[k][0] = 0;
    }

    Params pampam = {a, &f, tau, h};
    four_points_next_layer(grid_solution[0], grid_solution[1], pampam);
    for(int j = 2; j < M_tau; j++){
        cross_next_layer(grid_solution, j, pampam);
    }

    FILE* save = fopen("results.txt", "w");
    if (save == nullptr) {
        printf("Ошибка открытия файла!\n");
        return 1;
    }

    for (int k = 0; k < M_tau; ++k) {
        for (int n = 0; n < N_x; ++n) {
            fprintf(save, "%g ", grid_solution[k][n]);
        }
        fprintf(save, "\n");
    }

    fclose(save);
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
