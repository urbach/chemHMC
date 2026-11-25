import re
import numpy as np
import matplotlib.pyplot as plt

def get_data(filename):
  # Lists to store the data
  trajectories = []
  kinetic = []
  potential = []
  deltaE = []

  # Regular expression to match the trajectory lines
  pattern = re.compile(
    r"trajectory\s+(\d+):\s+"
    r"deltaE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

  with open(filename, "r") as f:
    for line in f:
      match = pattern.search(line)
      if match:
        trajectories.append(int(match.group(1)))
        deltaE.append(float(match.group(2)))

  return np.array(trajectories), np.array(deltaE)

def plot_data(dt, filename, title):
  trajectories, deltaE = get_data(filename)

  # Set up plot parameters
  plt.style.use("scientificLHPC.mplstyle")
  plt.rcParams.update({"axes.labelsize": 22, "ytick.labelsize": 18, "axes.titlesize": 24, "xtick.labelsize": 18, "legend.title_fontsize": 10})
  
  # Set up figure and axes
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 6.0))
  ax.set_axisbelow(True)

  ax.plot(trajectories, deltaE, color="cornflowerblue", linewidth=2.0, label=rf"$\textrm{{Mean: }} {np.mean(deltaE):.2e}, \textrm{{Std: }} {np.std(deltaE):.2e}$")
  ax.set_xlabel(r"$\textrm{Time, } t$")
  ax.set_ylabel(r"$\textrm{Energy difference, } \Delta E$")
  ax.set_title(title)

  ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=18)
  plt.tight_layout()
  plt.show()

dt_file_path = "Input_files/low_dens/Sweep_dt/dts_q1.in"
with open(dt_file_path, "r") as dt_file:
  for line in dt_file:
    dt = line.strip()
    data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/Data/low_dens/Sweep_dt/n2_q1/"
    filename = data_dir + f"argon_low_{dt}.xyz"
    plot_data(float(dt), filename, rf"$\textrm{{Low density, Leapfrog with }} dt = {dt}$")



dt = "0.00001"
data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/low_dens/leapfrog2/"
filename = data_dir + f"argon_dt-{dt}_leapfrog2.xyz"
#plot_data(float(dt), filename, rf"$\textrm{{Low density, Leapfrog with }} dt = {dt}$")

dt = "0.0001"
data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/low_dens/omelyan2/"
filename = data_dir + f"argon_dt-{dt}_omelyan2.xyz"
#plot_data(float(dt), filename, rf"$\textrm{{Low density, Omelyan2 with }} dt = {dt}$")

dt = "0.0001"
data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/low_dens/forest_ruth4/"
filename = data_dir + f"argon_dt-{dt}_forest_ruth4.xyz"
#plot_data(float(dt), filename, rf"$\textrm{{Low density, Forest-Ruth4 with }} dt = {dt}$")

dt = "0.0001"
data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/low_dens/OMF4/"
filename = data_dir + f"argon_dt-{dt}_OMF4.xyz"
#plot_data(float(dt), filename, rf"$\textrm{{Low density, OMF4 with }} dt = {dt}$")
