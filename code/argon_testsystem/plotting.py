import re
import numpy as np
import matplotlib.pyplot as plt

def get_data(filename):
  # Lists to store the data
  trajectories = []
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

def plot_single(dt, filename, title, savefile):
  trajectories, deltaE = get_data(filename)

  # Set up plot parameters
  plt.style.use("scientificLHPC.mplstyle")
  plt.rcParams.update({"axes.labelsize": 22, "ytick.labelsize": 18, "axes.titlesize": 24, "xtick.labelsize": 18, "legend.title_fontsize": 10})
  
  # Set up figure and axes
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 6.0))
  ax.set_axisbelow(True)

  ax.plot(trajectories, deltaE, color="cornflowerblue", linewidth=2.0, label=rf"$\textrm{{Mean: }} {np.mean(deltaE):.2e}, \textrm{{Std: }} {np.std(deltaE):.2e}$")
  ax.set_xlabel(r"$\textrm{Trajectories}$")
  ax.set_ylabel(r"$\textrm{Energy difference, } \Delta E$")
  ax.set_title(title)

  ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=18)
  plt.tight_layout()
  #plt.show()
  plt.savefig(savefile)

def plot_cost(schemes, savefile):

  # Set up plot parameters
  plt.style.use("scientificLHPC.mplstyle")
  plt.rcParams.update({"axes.labelsize": 22, "ytick.labelsize": 18, "axes.titlesize": 24, "xtick.labelsize": 18, "legend.title_fontsize": 10})
  
  # Set up figure and axes
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10.0, 6.0))
  ax.set_axisbelow(True)

  for scheme in schemes:
    dts = scheme["dts"]
    stdEs = np.zeros(len(dts))
    for i, dt in enumerate(dts):
      filename = scheme["data_dir"] + f"{dt}.xyz"
      trajectories, deltaE = get_data(filename)
      stdEs[i] = np.std(deltaE)

    q = scheme["cycles"]
    ax.plot(q/dts, stdEs, color=scheme["color"], linewidth=2.0, linestyle=scheme["style"], label=scheme["label"])
  
  ax.set_xlabel(r"$\textrm{Cost, } \frac{q}{dt}$")
  ax.set_ylabel(r"$\textrm{RMSE of } \Delta E$")
  ax.set_xscale("log")
  ax.set_yscale("log")

  ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=18)
  plt.tight_layout()
  #plt.show()
  plt.savefig(savefile)



dts = ["40.0", "35.0", "30.0", "25.0", "20.0", "15.0", "10.0", "8.0", "6.0", "4.0", "2.0", "1.0",
       "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1",
       "0.09", "0.08", "0.07", "0.06", "0.05", "0.04", "0.03", "0.02", "0.01",
       "0.009", "0.008", "0.007", "0.006", "0.005", "0.004", "0.003", "0.002", "0.001",
       "0.0009", "0.0008", "0.0007", "0.0006", "0.0005", "0.0004", "0.0003", "0.0002", "0.0001"]

for dt in dts:
  data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/Data/low_dens/Sweep_dt/OMF4/"
  filename = data_dir + f"argon_low_{dt}.xyz"
  #plot_single(float(dt), filename, rf"$\textrm{{Low density, Leapfrog with }} dt = {dt}$", f"Plots/low_dens/OMF4/dt-{dt}.pdf")

dts = np.array([25.0, 20.0, 15.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0,
                0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
                0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001,
                0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001])

directory = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/Data/low_dens/Sweep_dt/"

scheme_n2_q1 = {"dts": dts,
                "data_dir": directory + "n2_q1/argon_low_",
                "cycles": 1,
                "color": "crimson",
                "style": "solid",
                "label": r"$\textrm{Leapfrog: } n=2, q=1$"}

scheme_n2_q2 = {"dts": dts,
                "data_dir": directory + "n2_q2/argon_low_",
                "cycles": 2,
                "color": "crimson",
                "style": "dashed",
                "label": r"$\textrm{Omelyan \textit{et al.}: } n=2, q=2$"}

scheme_n4_q3 = {"dts": dts,
                "data_dir": directory + "n4_q3/argon_low_",
                "cycles": 3,
                "color": "darkviolet",
                "style": "solid",
                "label": r"$\textrm{Forest \& Ruth: } n=4, q=3$"}

scheme_OMF4 = {"dts": dts,
               "data_dir": directory + "OMF4/argon_low_",
               "cycles": 4,
               "color": "darkviolet",
               "style": "dashed",
               "label": r"$\textrm{Omelyan \textit{et al.}: } n=4$"}

schemes = [scheme_n2_q1, scheme_n2_q2, scheme_n4_q3, scheme_OMF4]

plot_cost(schemes, f"Plots/cost/low_dens/schemes.pdf")



dt = "0.00001"
data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/low_dens/leapfrog2/"
filename = data_dir + f"argon_dt-{dt}_leapfrog2.xyz"
#plot_single(float(dt), filename, rf"$\textrm{{Low density, Leapfrog with }} dt = {dt}$")

dt = "0.0001"
data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/low_dens/omelyan2/"
filename = data_dir + f"argon_dt-{dt}_omelyan2.xyz"
#plot_single(float(dt), filename, rf"$\textrm{{Low density, Omelyan2 with }} dt = {dt}$")

dt = "0.0001"
data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/low_dens/forest_ruth4/"
filename = data_dir + f"argon_dt-{dt}_forest_ruth4.xyz"
#plot_single(float(dt), filename, rf"$\textrm{{Low density, Forest-Ruth4 with }} dt = {dt}$")

dt = "0.0001"
data_dir = "/home/marko/Faks/PhD/Projects/Symplectic_integrators/chemHMC/code/argon_testsystem/low_dens/OMF4/"
filename = data_dir + f"argon_dt-{dt}_OMF4.xyz"
#plot_single(float(dt), filename, rf"$\textrm{{Low density, OMF4 with }} dt = {dt}$")
