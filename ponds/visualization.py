import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature  # type: ignore
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import cm

from typing import Any
from numpy.typing import NDArray

# from build.lib.build.lib.ponds.core import PONDS
# import ponds


###################################################################################################
############################################ Plot Maps ############################################
###################################################################################################

def worldmap(
    ponds: Any,
    ax: Axes | None = None,
    verbose: bool = False,
    figsize: tuple[float, float] = (16, 8),
    coastline: bool = True,
    borders: bool = True,
    show_cells: bool = False,
    shift_type: bool = False,
    title: str = "Cluster Masks",
) -> tuple[Figure, Axes]:
    """Plot the PONDS data on a world map using lat/lon as axis ticks.

    Args:
        ponds: PONDS dataset containing cluster masks.
        verbose: If True, print additional information during processing.
        figsize: Size of the figure to create.
        coastline: If True, add coastline features to the map.
        borders: If True, add country borders to the map.
        show_cells: If True, draw grid lines based on xarray resolution.
        title: Title for the plot.
    """
    # Create figure/axes only if not provided
    if ax is None:
        # Set up map with Cartopy
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        figsize = None # type: ignore
        fig = ax.figure

    # Extract cluster masks
    cluster_masks: list[NDArray[np.float64]] = []
    if verbose:
        print("Extracting cluster masks...")
    cluster_key = ponds.data.attrs["cluster_key"]
    for name, da in ponds.data.data_vars.items():
        if name.startswith(cluster_key):
            if verbose:
                print(f"Found cluster variable: {name}")
            cluster_masks.append(da)
    if verbose:
        print(f"Found {len(cluster_masks)} cluster variables.")

    # Specify Colors
    cland = "#fff3e0"
    cocean = "#e0f7fa"
    cblobs = [
        "#d36060",  # soft red
        "#f0a830",  # warm orange
        "#f6d55c",  # pastel yellow
        "#7bc043",  # fresh green
        "#69d2e7",  # sky blue
        "#6384b3",  # dusty blue
        "#f17cb0",  # pink rose
        "#c894d5",  # light purple
        "#ff9671",  # coral
        "#52b788",  # soft teal
        "#6c5ce7",  # indigo
    ]

    # Add map features
    if borders:
        ax.add_feature(cfeature.BORDERS, linestyle=":")  # type: ignore
        ax.add_feature(cfeature.LAND, facecolor=cland)  # type: ignore
        ax.add_feature(cfeature.OCEAN, facecolor=cocean)  # type: ignore
    if coastline:
        ax.add_feature(cfeature.COASTLINE)  # type: ignore
        ax.add_feature(cfeature.LAND, facecolor=cland)  # type: ignore
        ax.add_feature(cfeature.OCEAN, facecolor=cocean)  # type: ignore
    if not show_cells:
        ax.gridlines(  # type: ignore
            draw_labels=True,
            colors="gray",
            linestyle="-.",
        )
    else:
        ax.gridlines(draw_labels=True, color="none")  # type: ignore

    for i, mask in enumerate(cluster_masks):
        collapsed = mask.any(dim="time").astype(int)  # type: ignore

        # Get lat/lon coordinates safely
        lat = collapsed.coords["lat"].values  # type: ignore
        lon = collapsed.coords["lon"].values  # type: ignore

        # Define image extent from lat/lon
        extent: list[float] = [lon.min(), lon.max(), lat.min(), lat.max()]  # type: ignore

        # Plot the mask on the map
        ax.imshow(  # type: ignore
            collapsed,  # type: ignore
            cmap=ListedColormap(["none", cblobs[i % len(cblobs)]]),
            alpha=0.6,
            extent=extent,  # type: ignore
            origin="lower",
            transform=ccrs.PlateCarree(),
        )

        # Add contour outlines (white)
        ax.contour(  # type: ignore
            lon,  # type: ignore
            lat,  # type: ignore
            collapsed.astype(int),  # type: ignore
            levels=[0.5],
            colors=cblobs[i % len(cblobs)],
            linewidths=2,
            transform=ccrs.PlateCarree(),
        )

    # Add optional grid lines based on lat/lon spacing
    if show_cells:
        lat = ponds.data.coords["lat"].values  # type: ignore
        lon = ponds.data.coords["lon"].values  # type: ignore

        if verbose:
            print("Drawing grid lines based on xarray resolution...")
        # Compute midpoints and deltas
        lat_edges = _get_grid_edges(lat)  # type: ignore
        lon_edges = _get_grid_edges(lon)  # type: ignore

        for x in lon_edges:
            ax.plot(  # type: ignore
                [x, x],
                [lat_edges[0], lat_edges[-1]],
                color="black",
                linewidth=0.5,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )

        for y in lat_edges:
            ax.plot(  # type: ignore
                [lon_edges[0], lon_edges[-1]],
                [y, y],
                color="black",
                linewidth=0.5,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )

    # Add optional shift type in the center of the cluster
    if shift_type:
        if verbose:
            print("Adding shift type labels...")
        for i, mask in enumerate(cluster_masks):
            # Get the center of the mask
            center_lat = mask.attrs["center_lat"]  # type: ignore
            center_lon = mask.attrs["center_lon"]  # type: ignore
            shift_type = str(mask.attrs["shift_type"]).split(".")[-1][:-2]  # type: ignore
            if verbose:
                print(f"Shift type for mask {i + 1}: {shift_type}")

            # Add text label at the center with a non-transparent box
            ax.text(  # type: ignore
                center_lon,  # type: ignore
                center_lat,  # type: ignore
                f"{shift_type}",
                fontsize=10,
                color="black",
                ha="center",
                va="center",
                transform=ccrs.PlateCarree(),
                bbox=dict(
                    facecolor=cblobs[i % len(cblobs)],
                    edgecolor="none",
                    alpha=1.0,
                    boxstyle="round,pad=0.3",
                ),
            )

    ax.set_title(title)  # type: ignore

    return fig, ax


def _get_grid_edges(
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Helper to calculate grid edges from 1D coordinate centers."""

    # Handle trivial case of a single coordinate
    if coords.size < 2:
        return coords.copy()

    # Calculate edge positions
    step = (coords[-1] - coords[0]) / (len(coords))
    edges: NDArray[np.float64] = np.arange(
        coords[0], coords[-1] + step, step, dtype=np.float64
    )
    return edges


###################################################################################################
######################################## Plot Time Series #########################################
###################################################################################################


def plot_single_ts(
    ponds: Any,
    lat: float,
    lon: float,
    alpha: float = 0.5,
    linewidth: float = 0.5,
    figsize: tuple[float, float] = (16, 4),
    title: str = "Time Series for Single Cell",
) -> tuple[Figure, Axes]:
    """
    Plots time series of some exemplary grid cell from the PONDS dataset.
    """

    fig, ax = plt.subplots(  # type: ignore
        figsize=figsize,
    )

    # find nearest grid cell
    latitudes = ponds.data.coords["lat"].values  # type: ignore
    longitudes = ponds.data.coords["lon"].values  # type: ignore
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()

    ax.plot(  # type: ignore
        ponds.data.ts[:, lat_idx, lon_idx],
        linestyle="-",
        linewidth=linewidth,
        marker=None,
        color="grey",
        alpha=alpha,
    )

    # eyecandy
    ax.set_title(title + f"\n(Lat: {lat}, Lon: {lon})")  # type: ignore
    ax.set_xlabel("Time")  # type: ignore
    ax.set_ylabel("Value")  # type: ignore

    return fig, ax


def plot_some_ts(
    ponds: Any,
    ratio: float = 0.1,
    alpha: float = 0.5,
    linewidth: float = 0.5,
    figsize: tuple[float, float] = (16, 4),
    title: str = "Time Series for Some Example Cells",
    fontsize: int = 12,
) -> tuple[Figure, Axes]:
    """
    Plots time series of some exemplary grid cell from the PONDS dataset.
    """

    fig, ax = plt.subplots(  # type: ignore
        figsize=figsize,
    )

    ts = ponds.data.ts  # shape: (time, lat, lon)
    # stack spatial dimensions (no copy!)
    ts_stacked = ts.stack(cell=("lat", "lon"))

    n_cells = ts_stacked.sizes["cell"]
    n_plot = max(1, int(n_cells * ratio))

    # randomly sample cells
    cell_idx = np.random.choice(n_cells, size=n_plot, replace=False)

    # plot in chunks for further speedup ( -> ---.values is expensive for large arrays)
    for chunk in np.array_split(cell_idx, 10):
        ax.plot(
            ts_stacked.isel(cell=chunk).values,
            color="grey",
            alpha=alpha,
            linewidth=linewidth,
            )

    # eyecandy
    ax.set_title(title, fontsize=fontsize)  # type: ignore
    # - set x-axis
    ax.set_xlabel("Time", fontsize=fontsize)  # type: ignore
    x_ticks = np.arange(
        0,
        ponds.data.sizes["time"]+1,     # +1 to include last tick
        step=max(1, ponds.data.sizes["time"] // 10)) # type: ignore
    ax.set_xticks(x_ticks)  # type: ignore
    ax.set_xticklabels(np.round(x_ticks,5), fontsize=fontsize)  # type: ignore
    ax.set_xlim(-1, ponds.data.sizes["time"])  # type: ignore
    # - set y-axis
    ax.set_ylabel("Observable Value", fontsize=fontsize)  # type: ignore
    ax.set_yticks(ax.get_yticks())  # type: ignore
    ax.set_yticklabels(np.round(ax.get_yticks(),5), fontsize=fontsize)  # type: ignore

    return fig, ax


def plot_all_ts(
    ponds: Any,
    alpha: float = 0.5,
    linewidth: float = 0.5,
    figsize: tuple[float, float] = (16, 4),
    title: str = "Time Series for All Cells",
) -> tuple[Figure, Axes]:
    """
    Plots the time series of every grid cell from the PONDS dataset.
    """

    fig, ax = plt.subplots(  # type: ignore
        figsize=figsize,
    )

    for i in range(ponds.data.sizes["lat"]):
        for j in range(ponds.data.sizes["lon"]):
            ax.plot(  # type: ignore
                ponds.data.ts[:, i, j],
                linestyle="-",
                linewidth=linewidth,
                marker=None,
                color="grey",
                alpha=alpha,
            )

    # eyecandy
    ax.set_title(title)  # type: ignore
    ax.set_xlabel("Time")  # type: ignore
    ax.set_ylabel("Value")  # type: ignore

    return fig, ax


###################################################################################################
####################################### Plot Correlations #########################################
###################################################################################################


def barplot_3D(
    mask: NDArray[np.bool_],
    xarray: NDArray[np.float64],
    yarray: NDArray[np.float64],
    time_values: NDArray[np.float64],
    alpha: float = 1.0,
    cmap: str = "viridis",
    linewidth: float = 0.3,
    elev: float = 40.0,
    azim: float = 120.0,
) -> tuple[Figure, Axes]:
    """
    Create an interactive 3D bar plot of values on a masked grid.
    Bars are colored based on their height using a colormap.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Create grid
    x, y = np.meshgrid(xarray, yarray)
    z_base = time_values[mask].min() - 0.5
    z = np.full_like(time_values, z_base)
    dx = np.abs(xarray[1] - xarray[0])
    dy = np.abs(yarray[1] - yarray[0])
    dz = time_values[mask] - z_base

    # Normalize values to [0,1] for colormap
    norm = plt.Normalize(time_values[mask].min(), time_values[mask].max())
    colors = plt.colormaps[cmap](norm(time_values[mask]))  # updated line

    # Plot bars with color-mapped heights
    ax.bar3d(
        x[mask],
        y[mask],
        z[mask],
        dx,
        dy,
        dz,  # time_values[mask],
        zsort="max",
        alpha=alpha,
        color=colors,
        edgecolor="black",
        linewidth=linewidth,
    )

    # eye candy
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    ax.set_zlim(z_base, time_values[mask].max() + 0.1)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_zlabel("Shift Time Offset [time steps]")
    if np.abs(elev) >= 80:
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.set_zlabel("")

    # Add a colorbar
    mappable = cm.ScalarMappable(cmap=plt.colormaps[cmap], norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        shrink=0.5,
        aspect=10,
        pad=0.1,
        label="Shift Time Offset [time steps]",
    )
    # Set colorbar ticks to integers only
    cbar.set_ticks(
        np.arange(int(time_values[mask].min()), int(time_values[mask].max()) + 1, 2)
    )
    cbar.ax.set_yticklabels([str(int(t)) for t in cbar.get_ticks()])

    # Add an inset axes for a head-up view
    inset_ax = fig.add_axes([0.65, 0.65, 0.3, 0.3], projection="3d")
    inset_ax.set_position([0.05, 0.65, 0.3, 0.3])
    inset_ax.set_facecolor((0, 0, 0, 0))

    inset_ax.bar3d(
        x[mask],
        y[mask],
        z[mask],
        dx,
        dy,
        dz,
        zsort="max",
        alpha=alpha,
        color=colors,
        edgecolor="black",
        linewidth=linewidth,
    )

    # Set a custom view angle for the inset
    inset_ax.view_init(elev=-90, azim=-90)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title("Bird View", fontsize=8)

    return fig, ax
