# kml-to-gmaps

## Disclaimer

> [!WARNING]
> This code was generated in part by an AI assistant. While I have reviewed and modified it, I make no claims of exclusive ownership. This repository is provided "as-is" without any warranty or guarantee of originality or suitability for any purpose. Users should conduct their own due diligence before use.

## About this repository

This repository contains a script to convert KML (Keyhole Markup Language) files to Google Maps URLs with the aim of allowing navigation along the route with Google Maps as best as possible.

### Why Use This?

While Google Maps offers the option to generate custom routes with the [My Maps](https://www.google.com/maps/d/u/0/) feature, Google Maps does not by default allow you to follow along that exact route. While other mapping solutions exist that allow you to follow along exact routes like [OsmAnd](https://osmand.net/) or [Outdooractive](https://www.outdooractive.com/) they fall short in other areas such as traffic statistics or rerouting in case of construction work.

In addition to that, Google Maps has a limitation of 9 waypoints per route, which makes it difficult to set up longer trips with a specific paths manually.

## Table of Contents

- [kml-to-gmaps](#kml-to-gmaps)
  - [Disclaimer](#disclaimer)
  - [About this repository](#about-this-repository)
    - [Why Use This?](#why-use-this)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
    - [Installation \& Basic Usage](#installation--basic-usage)
    - [Advanced Examples](#advanced-examples)
  - [Supported Parameters](#supported-parameters)

## Features

- Converts KML to Google Maps URLs
- Customizable waypoint amount
- Route "splitting" with more than 9 Waypoints
- Two waypoint selection algorithms
  - RDP ([Ramer–Douglas–Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm))
  - Even distribution
- No external dependencies

## Prerequisites

- Python >= 3.11.x

## Usage

### Installation & Basic Usage

1. Either Download `kml-to-gmaps.py` or clone the repository using `git clone https://github.com/RotusMaximus/kml-to-gmaps`
2. Open your Terminal
3. Navigate to the folder you placed the script in
4. Call the script like so `python kml_to_gmaps.py <KML_FILE_NAME_OR_PATH>.kml`
5. Done! You should see the output URL in your Terminal window

### Advanced Examples

Using five waypoints using even distribution:

```python
python kml_to_gmaps.py route.kml -n 5 -m even
```

Using 18 waypoints using a cycling route:

```python
python kml_to_gmaps.py route.kml -n 18 -t bicycling
```

## Supported Parameters

| Parameter              | Description                                                                                            | Default   | Possible Values                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------ | --------- | ------------------------------------------------------- |
| `-h` `--help`          | Shows the help message.                                                                                | -         |                                                         |
| `-n` `--num_waypoints` | Total number of intermediate waypoints desired for the route. If > 9, multiple URLs will be generated. | 9         | Any Number                                              |
| `-m` `--method`        | Waypoint selection method: 'even' (evenly distributed) or 'rdp' (significant points).                  | `rdp`     | `rdp`, `even`                                           |
| `-t` `--travelmode`    | Google Maps travel mode.                                                                               | `driving` | `driving`,`walking`,`bicycling`,`two-wheeler`,`transit` |
| `-o` `--output`        | Optional file path to save the generated URLs (one URL per line).                                      | -         | Absolute or relative file path                          |
