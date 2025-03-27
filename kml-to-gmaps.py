#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import urllib.parse
import math
import heapq
import re
import argparse  # Import argparse
import sys
import os  # To check if file exists

# --- Constants ---
GOOGLE_MAPS_WAYPOINT_LIMIT = 9  # Max intermediate waypoints per URL

# --- Helper Function for Distance Calculation (for RDP) ---


def point_segment_distance(p, a, b):
    """
    Calculates the perpendicular Euclidean distance of point p from line segment ab.
    Uses an approximation for lat/lon, scaling longitude by cos(latitude).

    Args:
        p (tuple): (latitude, longitude) of the point.
        a (tuple): (latitude, longitude) of the start of the segment.
        b (tuple): (latitude, longitude) of the end of the segment.

    Returns:
        float: The perpendicular distance. Returns 0 if segment has zero length.
    """
    lat_p, lon_p = p
    lat_a, lon_a = a
    lat_b, lon_b = b

    # Check if segment has near-zero length to avoid division issues
    if abs(lat_a - lat_b) < 1e-9 and abs(lon_a - lon_b) < 1e-9:
        # Calculate distance from p to point a (since a and b are virtually the same)
        avg_lat_rad = math.radians(lat_a)
        cos_avg_lat = math.cos(avg_lat_rad)
        dy_p = lat_p - lat_a
        dx_p = (lon_p - lon_a) * cos_avg_lat
        return math.sqrt(dx_p * dx_p + dy_p * dy_p)

    # Use average latitude for longitude scaling (approximation)
    avg_lat_rad = math.radians((lat_a + lat_b) / 2.0)
    cos_avg_lat = math.cos(avg_lat_rad)

    # Treat lat/lon as y/x, scaling x (longitude)
    y_p, x_p = lat_p, lon_p * cos_avg_lat
    y_a, x_a = lat_a, lon_a * cos_avg_lat
    y_b, x_b = lat_b, lon_b * cos_avg_lat

    # Segment vector components
    dx = x_b - x_a
    dy = y_b - y_a

    seg_sq_len = dx * dx + dy * dy
    # seg_sq_len should not be zero here due to the earlier check, but being safe:
    if seg_sq_len == 0:
        return 0.0

    # Project p onto the line ab
    t = ((x_p - x_a) * dx + (y_p - y_a) * dy) / seg_sq_len

    if t < 0:  # Projection falls "before" a
        proj_x, proj_y = x_a, y_a
    elif t > 1:  # Projection falls "after" b
        proj_x, proj_y = x_b, y_b
    else:  # Projection falls onto the segment
        proj_x = x_a + t * dx
        proj_y = y_a + t * dy

    # Distance from p to the projection point
    dist_x = x_p - proj_x
    dist_y = y_p - proj_y
    return math.sqrt(dist_x * dist_x + dist_y * dist_y)


def find_max_distance_point_index(points, start_idx, end_idx):
    """ Finds the index of the point with max distance from segment points[start_idx] to points[end_idx]. """
    max_d = 0.0
    max_idx = -1
    if start_idx >= end_idx - 1:  # Segment has no intermediate points
        return max_idx, max_d

    a = points[start_idx]
    b = points[end_idx]

    for i in range(start_idx + 1, end_idx):
        d = point_segment_distance(points[i], a, b)
        if d > max_d:
            max_d = d
            max_idx = i
    return max_idx, max_d

# --- Waypoint Selection Methods ---


def select_waypoints_rdp(coords, num_waypoints_to_select):
    """
    Selects start, end, and a specified number of waypoints based on RDP-like significance.

    Args:
        coords (list): A list of (latitude, longitude) tuples for the whole route.
        num_waypoints_to_select (int): The total number of intermediate waypoints desired.

    Returns:
        tuple: (origin, waypoints_list, destination) or None if input is invalid.
               waypoints_list contains the selected intermediate points, sorted by original route order.
    """
    n = len(coords)
    if n < 2:
        print("Error: Coordinate list must have at least a start and end point.")
        return None

    origin = coords[0]
    destination = coords[-1]
    num_intermediate_coords = n - 2

    if num_intermediate_coords <= 0:
        return origin, [], destination
    if num_waypoints_to_select >= num_intermediate_coords:
        # print(f"Warning: Requested {num_waypoints_to_select} waypoints, but only {num_intermediate_coords} intermediate points available. Using all.")
        return origin, coords[1:-1], destination

    selected_indices = {0, n - 1}
    pq = []
    max_idx, max_d = find_max_distance_point_index(coords, 0, n - 1)
    if max_idx != -1:
        heapq.heappush(pq, (-max_d, max_idx, 0, n - 1))

    while pq and len(selected_indices) < num_waypoints_to_select + 2:
        neg_dist, current_idx, start_idx, end_idx = heapq.heappop(pq)
        if current_idx in selected_indices:
            continue
        selected_indices.add(current_idx)
        max_idx1, max_d1 = find_max_distance_point_index(
            coords, start_idx, current_idx)
        if max_idx1 != -1:
            heapq.heappush(pq, (-max_d1, max_idx1, start_idx, current_idx))
        max_idx2, max_d2 = find_max_distance_point_index(
            coords, current_idx, end_idx)
        if max_idx2 != -1:
            heapq.heappush(pq, (-max_d2, max_idx2, current_idx, end_idx))

    waypoint_indices = sorted(list(selected_indices - {0, n - 1}))
    waypoints = [coords[i] for i in waypoint_indices]
    return origin, waypoints, destination


def select_waypoints_even(coords, num_waypoints_to_select):
    """
    Selects start, end, and a specified number of evenly distributed waypoints.

    Args:
        coords (list): A list of (latitude, longitude) tuples for the whole route.
        num_waypoints_to_select (int): The total number of intermediate waypoints desired.

    Returns:
        tuple: (origin, waypoints_list, destination) or None if input is invalid.
    """
    n = len(coords)
    if n < 2:
        print("Error: Coordinate list must have at least a start and end point.")
        return None

    origin = coords[0]
    destination = coords[-1]
    intermediate_points = coords[1:-1]
    num_intermediate = len(intermediate_points)

    if num_intermediate <= 0:
        return origin, [], destination
    if num_waypoints_to_select >= num_intermediate:
        # print(f"Warning: Requested {num_waypoints_to_select} waypoints, but only {num_intermediate} intermediate points available. Using all.")
        return origin, intermediate_points, destination

    waypoints = []
    num_segments = num_waypoints_to_select + 1
    potential_indices = set()
    for i in range(1, num_segments):
        ideal_index = (i * (num_intermediate + 1) / num_segments) - 1
        index = max(0, min(num_intermediate - 1, round(ideal_index)))
        potential_indices.add(index)

    sorted_unique_indices = sorted(list(potential_indices))
    needed = num_waypoints_to_select - len(sorted_unique_indices)
    idx_ptr = 0
    while needed > 0 and idx_ptr < len(sorted_unique_indices) - 1:
        current_idx = sorted_unique_indices[idx_ptr]
        next_idx = sorted_unique_indices[idx_ptr + 1]
        mid_point_idx = (current_idx + next_idx) // 2
        if mid_point_idx > current_idx and mid_point_idx < next_idx and mid_point_idx not in potential_indices:
            potential_indices.add(mid_point_idx)
            needed -= 1
        idx_ptr += 1

    final_indices = sorted(list(potential_indices))[:num_waypoints_to_select]
    waypoints = [intermediate_points[i] for i in final_indices]
    return origin, waypoints, destination

# --- KML Parsing (robust version) ---


def extract_coordinates_from_kml(kml_file_path):
    try:
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        tree = ET.parse(kml_file_path)
        root = tree.getroot()
        coord_elements = root.findall(
            './/kml:LineString/kml:coordinates', namespaces)

        if not coord_elements:
            coord_elements = root.findall('.//LineString/coordinates')
            if not coord_elements:
                print(
                    f"Error: No <coordinates> tag found within a <LineString> in the KML.")
                return None

        coord_string = coord_elements[0].text
        if not coord_string:
            print(f"Error: <coordinates> tag is empty in the KML.")
            return None

        coord_pairs_str = re.split(r'\s+', coord_string.strip())
        coordinates = []
        for pair_str in coord_pairs_str:
            if not pair_str:
                continue
            try:
                parts = pair_str.split(',')
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    coordinates.append((lat, lon))
                else:
                    print(
                        f"Warning: Skipping invalid coordinate pair string: {pair_str}")
            except ValueError:
                print(
                    f"Warning: Skipping non-numeric coordinate pair string: {pair_str}")

        if len(coordinates) < 2:
            print("Error: Need at least two coordinates (start and end) in the KML.")
            return None
        return coordinates

    except ET.ParseError as e:
        print(f"Error parsing KML: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: KML file not found at path: {kml_file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during KML parsing: {e}")
        return None

# --- Google Maps URL Creation (identical) ---


def create_google_maps_url(origin, waypoints, destination, travelmode='driving'):
    base_url = "https://www.google.com/maps/dir/?api=1"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "travelmode": travelmode
    }
    if waypoints:
        waypoints_str = "|".join([f"{lat},{lon}" for lat, lon in waypoints])
        params["waypoints"] = waypoints_str
    encoded_params = urllib.parse.urlencode(
        params, quote_via=urllib.parse.quote)
    return f"{base_url}&{encoded_params}"

# --- Main Execution Logic ---


def main():
    parser = argparse.ArgumentParser(
        description="Convert KML route files to Google Maps URLs with waypoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help
    )

    parser.add_argument("kml_file", help="Path to the input KML file.")
    parser.add_argument("-n", "--num_waypoints", type=int, default=GOOGLE_MAPS_WAYPOINT_LIMIT,
                        help=f"Total number of intermediate waypoints desired for the route. "
                             f"If > {GOOGLE_MAPS_WAYPOINT_LIMIT}, multiple URLs will be generated.")
    parser.add_argument("-m", "--method", choices=['even', 'rdp'], default='rdp',
                        help="Waypoint selection method: 'even' (evenly distributed) or 'rdp' (significant points).")
    parser.add_argument("-t", "--travelmode", choices=['driving', 'walking', 'bicycling', 'two-wheeler', 'transit'],
                        default='driving', help="Google Maps travel mode.")
    parser.add_argument(
        "-o", "--output", help="Optional file path to save the generated URLs (one URL per line).")

    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.exists(args.kml_file):
        print(f"Error: Input KML file not found: {args.kml_file}")
        sys.exit(1)
    if args.num_waypoints < 0:
        print(
            f"Error: Number of waypoints ({args.num_waypoints}) cannot be negative.")
        sys.exit(1)

    # --- Processing ---
    print(f"Processing KML file: {args.kml_file}")
    print(
        f"Requested {args.num_waypoints} total intermediate waypoints using '{args.method}' method.")
    print(f"Travel mode set to: {args.travelmode}")

    coordinates = extract_coordinates_from_kml(args.kml_file)

    if not coordinates:
        print("Exiting due to coordinate extraction failure.")
        sys.exit(1)

    print(f"Extracted {len(coordinates)} coordinates from KML.")

    num_intermediate_available = len(coordinates) - 2
    actual_waypoints_to_select = args.num_waypoints

    if len(coordinates) <= 2:
        print("Warning: Route only has start/end points. Cannot select intermediate waypoints.")
        actual_waypoints_to_select = 0
        selected_waypoints = []
        origin = coordinates[0]
        destination = coordinates[-1]
    elif args.num_waypoints == 0:
        print("0 waypoints requested. Generating URL with only start and end.")
        selected_waypoints = []
        origin = coordinates[0]
        destination = coordinates[-1]
    elif args.num_waypoints > num_intermediate_available:
        print(
            f"Warning: Requested {args.num_waypoints} waypoints, but only {num_intermediate_available} intermediate points available. Using all {num_intermediate_available}.")
        actual_waypoints_to_select = num_intermediate_available
        # Fall through to selection method, which will now use all points

    # Select the total set of waypoints using the chosen method
    if actual_waypoints_to_select > 0:
        if args.method == 'rdp':
            selection_result = select_waypoints_rdp(
                coordinates, actual_waypoints_to_select)
            method_name = "RDP"
        elif args.method == 'even':
            selection_result = select_waypoints_even(
                coordinates, actual_waypoints_to_select)
            method_name = "Even Distribution"
        else:  # Should be caught by argparse, but as a fallback
            print(f"Error: Invalid selection_method '{args.method}'.")
            sys.exit(1)

        if not selection_result:
            print(f"Exiting due to {args.method} waypoint selection failure.")
            sys.exit(1)

        origin, selected_waypoints, destination = selection_result
        print(
            f"Selected {len(selected_waypoints)} waypoints via {method_name}.")
    # else case (0 waypoints needed) handled above

    # --- URL Generation and Output ---
    route_points = [origin] + selected_waypoints + [destination]
    # How many were *actually* selected
    num_selected_intermediate = len(selected_waypoints)

    generated_urls = []

    if num_selected_intermediate <= GOOGLE_MAPS_WAYPOINT_LIMIT:
        # Only one URL needed
        url = create_google_maps_url(
            origin, selected_waypoints, destination, travelmode=args.travelmode)
        generated_urls.append(url)
        print("\nGenerated 1 Google Maps URL:")

    else:
        # Multiple URLs needed
        print(
            f"\nRequested {num_selected_intermediate} waypoints exceeds limit ({GOOGLE_MAPS_WAYPOINT_LIMIT}). Generating multiple URLs:")
        num_segments_needed = math.ceil(
            num_selected_intermediate / GOOGLE_MAPS_WAYPOINT_LIMIT)

        for i in range(num_segments_needed):
            # Indices within the selected_waypoints list
            waypoint_start_idx = i * GOOGLE_MAPS_WAYPOINT_LIMIT
            waypoint_end_idx = min(
                waypoint_start_idx + GOOGLE_MAPS_WAYPOINT_LIMIT, num_selected_intermediate)

            # Determine origin and destination for this segment using route_points
            # Index in selected_waypoints corresponds to index+1 in route_points
            segment_origin_index = waypoint_start_idx
            segment_origin = route_points[segment_origin_index]

            # Index in selected_waypoints + 1 for destination in route_points
            segment_destination_index = waypoint_end_idx + 1
            segment_destination = route_points[segment_destination_index]

            # Extract waypoints for this specific URL segment
            segment_waypoints = selected_waypoints[waypoint_start_idx: waypoint_end_idx]

            url = create_google_maps_url(
                segment_origin, segment_waypoints, segment_destination, travelmode=args.travelmode)
            generated_urls.append(url)

    # --- Output URLs ---
    if args.output:
        try:
            with open(args.output, 'w') as f:
                for i, url in enumerate(generated_urls):
                    f.write(f"{url}\n")  # Write each URL on a new line
            print(
                f"\nSuccessfully saved {len(generated_urls)} URL(s) to {args.output}")
        except IOError as e:
            print(f"\nError writing to output file {args.output}: {e}")
            print("Printing URLs to console instead:")
            for i, url in enumerate(generated_urls):
                print(f"URL {i+1}: {url}")
    else:
        # Print to console if no output file specified
        print("\n--- Generated Google Maps URL(s) ---")
        if not generated_urls:
            print(
                "No URL generated (likely only start/end points in KML and 0 waypoints requested).")
        for i, url in enumerate(generated_urls):
            print(f"URL {i+1}: {url}")
        print("--- End of URLs ---")


if __name__ == "__main__":
    main()
