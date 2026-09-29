"""Build a static MIGN HEALPix graph for a fixed station coordinate set.

This follows the graph topology in the official MIGN
``step3_generate_graph_dgl_multi_step.py`` preprocessor, but omits daily
observations because TSLib injects each batch into the graph at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dgl
import healpy as hp
import numpy as np
import torch


def haversine(
    source_lon: np.ndarray,
    source_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    """Pairwise great-circle distance in thousands of kilometres."""
    source_lon, source_lat, target_lon, target_lat = map(
        np.radians, (source_lon, source_lat, target_lon, target_lat)
    )
    delta_lon = target_lon - source_lon[:, None]
    delta_lat = target_lat - source_lat[:, None]
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(source_lat[:, None])
        * np.cos(target_lat)
        * np.sin(delta_lon / 2) ** 2
    )
    return 6.371 * 2 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def build_graph(
    coordinates: np.ndarray,
    feature: str,
    input_length: int,
    output_length: int,
    refinement_level: int,
    neighbors: int,
):
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("station coordinates must have shape [N, >=2] (lat, lon, ...)")
    station_lat = coordinates[:, 0].astype(np.float64)
    station_lon = coordinates[:, 1].astype(np.float64)
    station_count = coordinates.shape[0]

    nside = 2**refinement_level
    pixel_count = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(pixel_count))
    pixel_lat = 90.0 - np.degrees(theta)
    pixel_lon = np.degrees(phi) - 180.0

    station_to_pixel = haversine(station_lon, station_lat, pixel_lon, pixel_lat)
    pixel_to_pixel = haversine(pixel_lon, pixel_lat, pixel_lon, pixel_lat)
    station_k = min(neighbors, station_count)
    pixel_k = min(neighbors, pixel_count)

    # Official encoder topology: for every mesh point, connect its k nearest
    # stations.  Decoder topology: connect every station to its k nearest mesh
    # points.  The same static topology is repeated for every forecast step.
    nearest_stations = np.argsort(station_to_pixel, axis=0)[:station_k].T
    encoder_src = nearest_stations.reshape(-1)
    encoder_dst = np.repeat(np.arange(pixel_count), station_k)

    nearest_pixels = np.argsort(station_to_pixel, axis=1)[:, :pixel_k]
    decoder_src = nearest_pixels.reshape(-1)
    decoder_dst = np.repeat(np.arange(station_count), pixel_k)

    mesh_neighbors = np.argsort(pixel_to_pixel, axis=1)[:, : pixel_k + 1]
    mesh_src = np.repeat(np.arange(pixel_count), pixel_k + 1)
    mesh_dst = mesh_neighbors.reshape(-1)

    edges = {}
    distances = {}
    for step in range(input_length):
        etype = (feature, f"t{step}_to_healpix", "healpix")
        edges[etype] = (encoder_src, encoder_dst)
        distances[etype] = station_to_pixel[encoder_src, encoder_dst]
    for step in range(output_length):
        etype = ("healpix", f"t{input_length + step}_to_{feature}", feature)
        edges[etype] = (decoder_src, decoder_dst)
        distances[etype] = station_to_pixel[decoder_dst, decoder_src]
    mesh_etype = ("healpix", "healpix_message", "healpix")
    edges[mesh_etype] = (mesh_src, mesh_dst)
    distances[mesh_etype] = pixel_to_pixel[mesh_src, mesh_dst]

    tensor_edges = {
        key: (torch.as_tensor(src), torch.as_tensor(dst))
        for key, (src, dst) in edges.items()
    }
    graph = dgl.heterograph(
        tensor_edges,
        num_nodes_dict={feature: station_count, "healpix": pixel_count},
    )
    for etype, distance in distances.items():
        graph.edges[etype].data["distance"] = torch.as_tensor(
            distance, dtype=torch.float32
        )

    total_length = input_length + output_length
    station_zeros = torch.zeros(station_count, 1)
    pixel_zeros = torch.zeros(pixel_count, 1)
    for step in range(total_length):
        graph.nodes[feature].data[f"t{step}"] = station_zeros.clone()
        graph.nodes["healpix"].data[f"{feature}_t{step}"] = pixel_zeros.clone()
    graph.nodes[feature].data["latitude"] = torch.as_tensor(
        station_lat, dtype=torch.float32
    ).unsqueeze(-1)
    graph.nodes[feature].data["longitude"] = torch.as_tensor(
        station_lon, dtype=torch.float32
    ).unsqueeze(-1)
    graph.nodes["healpix"].data["latitude"] = torch.as_tensor(
        pixel_lat, dtype=torch.float32
    ).unsqueeze(-1)
    graph.nodes["healpix"].data["longitude"] = torch.as_tensor(
        pixel_lon, dtype=torch.float32
    ).unsqueeze(-1)
    return graph, {feature: torch.ones(station_count, dtype=torch.bool)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature", default="MXSPD")
    parser.add_argument("--input-length", type=int, default=48)
    parser.add_argument("--output-length", type=int, default=72)
    parser.add_argument("--refinement-level", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=10)
    args = parser.parse_args()

    coordinates = np.load(args.coordinates)
    graph, labels = build_graph(
        coordinates=coordinates,
        feature=args.feature,
        input_length=args.input_length,
        output_length=args.output_length,
        refinement_level=args.refinement_level,
        neighbors=args.neighbors,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(output), [graph], labels)
    print(
        f"saved {output}: stations={graph.num_nodes(args.feature)}, "
        f"healpix={graph.num_nodes('healpix')}, edges={graph.num_edges()}"
    )


if __name__ == "__main__":
    main()
