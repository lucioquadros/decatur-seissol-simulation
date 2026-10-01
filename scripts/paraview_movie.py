"""Paraview movie frames of a SeisSol run using faults and wavefield output: 
   Faults colored by slip rate in a volume-rendered wavefield.

Run with ParaView's pvpython (e.g. on Windows) to render 0.6 s frame:
    pvpython.exe --force-offscreen-rendering paraview_movie.py D:\\decatur_showcase\\bob_will\\output --time 0.6
Frames go to <output>/../movie/frame_NNNN.png, then:
    ffmpeg -framerate 25 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 movie.mp4
"""

import argparse
import math
import os

from paraview.simple import (Calculator, CellDatatoPointData, ColorBy, Contour, GetColorTransferFunction,
                             GetOpacityTransferFunction, GetScalarBar, ResampleToImage,
                             SaveScreenshot, Show, Text, XDMFReader, CreateView)

BACKGROUND = [0.06, 0.06, 0.08]
SR_COLORS = [0.0, 0.42, 0.42, 0.46, 0.3, 0.9, 0.3, 0.05, 1.0, 1.0, 0.95, 0.45]
WAVE_COLORS = [0.0, 0.05, 0.20, 0.70, 0.5, 0.20, 0.55, 1.0, 1.0, 0.55, 0.85, 1.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", help="SeisSol output directory (decatur.xdmf, decatur-fault.xdmf, ...)")
    p.add_argument("--prefix", default="decatur")
    p.add_argument("--outdir", help="default: <output>/../movie")
    p.add_argument("--time", type=float, nargs="+", help="render only these times (s), for designing the scene")
    p.add_argument("--mode", choices=("volume", "contour"), default="volume")
    p.add_argument("--vmax", type=float, default=0.5, help="velocity magnitude at full color, m/s")
    p.add_argument("--srmax", type=float, default=3.0, help="slip rate at full color, m/s")
    p.add_argument("--fog", type=float, default=150.0,
                   help="volume opacity unit distance, m (larger is more transparent, default: %(default)s)")
    p.add_argument("--surface", action="store_true", help="add the free surface colored by vertical velocity")
    p.add_argument("--dims", type=int, nargs=3, default=(250, 280, 220),
                   help="resampling grid of the wavefield (default: %(default)s)")
    p.add_argument("--azimuth", type=float, default=200.0, help="camera direction from the faults, degrees from north")
    p.add_argument("--elevation", type=float, default=20.0, help="camera elevation, degrees")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--size", type=int, nargs=2, default=(1920, 1080))
    p.add_argument("--title", default="")
    return p.parse_args()


def reader(path, arrays):
    r = XDMFReader(FileNames=[path])
    r.CellArrayStatus = arrays
    r.UpdatePipelineInformation()
    return r


def colormap(name, rgb, vmin, vmax, opacity=None):
    """Fixed color (and opacity) range, set after ColorBy, which rescales to the first time step."""
    lut = GetColorTransferFunction(name)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RGBPoints = rgb
    lut.RescaleTransferFunction(vmin, vmax)
    pwf = GetOpacityTransferFunction(name)
    pwf.Points = opacity or [0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
    pwf.RescaleTransferFunction(vmin, vmax)
    return lut


def scalar_bar(lut, view, title, position):
    bar = GetScalarBar(lut, view)
    bar.Title, bar.ComponentTitle = title, ""
    bar.TitleColor = bar.LabelColor = [0.9, 0.9, 0.9]
    bar.TitleFontSize, bar.LabelFontSize = 16, 14
    bar.Orientation = "Horizontal"
    bar.WindowLocation = "Any Location"
    bar.Position, bar.ScalarBarLength = position, 0.22
    bar.RangeLabelFormat = "%-#.2g"
    bar.Visibility = 1


def main():
    args = parse_args()
    out = os.path.abspath(args.output)
    outdir = args.outdir or os.path.join(os.path.dirname(out), "movie")
    os.makedirs(outdir, exist_ok=True)

    view = CreateView("RenderView")
    view.ViewSize = list(args.size)
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = BACKGROUND

    fault = reader(os.path.join(out, f"{args.prefix}-fault.xdmf"), ["SRs", "SRd"])
    sr = Calculator(Input=fault, AttributeType="Cell Data", ResultArrayName="SR", Function="sqrt(SRs^2+SRd^2)")
    sr_points = CellDatatoPointData(Input=sr)
    fault_display = Show(sr_points, view)
    ColorBy(fault_display, ("POINTS", "SR"))
    # faults at rest stay faint, slipping parts opaque
    sr_lut = colormap("SR", SR_COLORS, 0.0, args.srmax, opacity=[0.0, 0.25, 0.5, 0.0, 0.05, 1.0, 0.5, 0.0,
                                                                 1.0, 1.0, 0.5, 0.0])
    sr_lut.EnableOpacityMapping = 1
    scalar_bar(sr_lut, view, "slip rate (m/s)", [0.05, 0.06])

    wave = reader(os.path.join(out, f"{args.prefix}.xdmf"), ["v1", "v2", "v3"])
    vmag = Calculator(Input=wave, AttributeType="Cell Data", ResultArrayName="vmag",
                      Function="sqrt(v1^2+v2^2+v3^2)")
    vmag_points = CellDatatoPointData(Input=vmag)
    if args.mode == "volume":
        grid = ResampleToImage(Input=vmag_points, SamplingDimensions=list(args.dims))
        wave_display = Show(grid, view)
        wave_display.SetRepresentationType("Volume")
        ColorBy(wave_display, ("POINTS", "vmag"))
        wave_display.ScalarOpacityUnitDistance = args.fog
    else:
        levels = [0.1 * args.vmax, 0.3 * args.vmax, 0.7 * args.vmax]
        shells = Contour(Input=vmag_points, ContourBy=["POINTS", "vmag"], Isosurfaces=levels)
        wave_display = Show(shells, view)
        ColorBy(wave_display, ("POINTS", "vmag"))
        wave_display.Opacity = 0.3
    # low amplitudes fully transparent, so the medium at rest disappears
    wave_lut = colormap("vmag", WAVE_COLORS, 0.0, args.vmax,
                        opacity=[0.0, 0.0, 0.5, 0.0, 0.1, 0.0, 0.5, 0.0, 0.4, 0.25, 0.5, 0.0, 1.0, 0.55, 0.5, 0.0])
    scalar_bar(wave_lut, view, "particle velocity (m/s)", [0.73, 0.06])

    if args.surface:
        surf = reader(os.path.join(out, f"{args.prefix}-surface.xdmf"), ["v3"])
        surf_display = Show(CellDatatoPointData(Input=surf), view)
        ColorBy(surf_display, ("POINTS", "v3"))
        surf_lut = GetColorTransferFunction("v3")
        surf_lut.ApplyPreset("Cool to Warm", True)
        surf_lut.AutomaticRescaleRangeMode = "Never"
        surf_lut.RescaleTransferFunction(-args.vmax, args.vmax)
        surf_display.Opacity = 0.6

    x0, x1, y0, y1, z0, z1 = fault.GetDataInformation().GetBounds()
    center = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
    radius = 0.5 * math.dist((x0, y0, z0), (x1, y1, z1))
    az, el = math.radians(args.azimuth), math.radians(args.elevation)
    direction = [math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)]
    distance = 3.2 * radius / args.zoom
    view.CameraFocalPoint = center
    view.CameraPosition = [c + distance * d for c, d in zip(center, direction)]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraViewAngle = 30.0

    label = Text()
    label_display = Show(label, view)
    label_display.WindowLocation = "Upper Left Corner"
    label_display.FontSize, label_display.Color = 22, [0.95, 0.95, 0.95]

    times = args.time or list(wave.TimestepValues)
    for i, t in enumerate(times):
        view.ViewTime = t
        label.Text = f"{args.title}\nt = {t:.2f} s" if args.title else f"t = {t:.2f} s"
        name = f"time_{t:.2f}s.png" if args.time else f"frame_{i:04d}.png"
        SaveScreenshot(os.path.join(outdir, name), view, ImageResolution=list(args.size))
        print(f"{name} ({i + 1}/{len(times)})", flush=True)


if __name__ == "__main__":
    main()
