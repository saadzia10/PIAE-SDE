# Physics-Informed Variational Autoencoder for Enhancing Data Quality to Improve the Forecasting Reliability of Carbon Dioxide Emissions from Agricultural Farms

### Authors: Corentin Houpert, Saad Zia (other names will be added here)

The main approach is implemented in the `pivae_sde` directory. 

### How to install dependencies:
#### Prerequisites: 
- Anaconda [https://docs.anaconda.com/anaconda/install/mac-os/]


#### Run the following command from the project root directory
```
conda env create -f env.yaml
```
The `env.yml` is found in the root directory of the project.

#### Experimentation
Head over to `pivae_sde` directory. All experiments are in the `PIVAE.ipynb` notebook. It is always kept updated. All experiments you perform from the notebook will have all data/training configurations and results (graphs and more) stored in a unique directory based on timestamps e.g., `pivae_sde/2024-8-27_16-18-2-282899`.

All required data files for experimentation are in the `pivae_sde/data_manipulation` directory.

#### Data Generation
All relevant code files for Data Partitioning (for $E_0$, $rb$, $\alpha$, $\beta$) and Manipulation are in the `pivae_sde/data_manipulation` directory.
