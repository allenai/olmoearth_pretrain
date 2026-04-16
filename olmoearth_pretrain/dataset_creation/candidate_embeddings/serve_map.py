"""Lightweight HTTP server for the embedding map + thumbnail images.

Usage:
    python serve_map.py --viz-dir /path/to/_viz
    python serve_map.py --viz-dir /path/to/_viz --thumbnail-dir /other/path/thumbs

The first form assumes thumbnails live inside ``_viz/thumbnails/``.
The second form creates a symlink so ``/thumbnails/`` resolves to a
directory stored elsewhere.

Cursor / VS Code will auto-forward the port so you can open the map
in your local browser.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import sys


def serve(
    viz_dir: str,
    port: int,
    thumbnail_dir: str | None = None,
    host: str = "127.0.0.1",
) -> None:
    """Serve a visualization directory and optional thumbnail directory over HTTP."""
    viz_dir = os.path.abspath(viz_dir)
    if not os.path.isdir(viz_dir):
        print(f"[error] --viz-dir does not exist: {viz_dir}", file=sys.stderr)
        sys.exit(1)

    # If thumbnails live outside _viz, symlink them in so relative URLs work.
    if thumbnail_dir:
        thumbnail_dir = os.path.abspath(thumbnail_dir)
        link_path = os.path.join(viz_dir, "thumbnails")
        if os.path.islink(link_path):
            os.unlink(link_path)
        if not os.path.exists(link_path):
            os.symlink(thumbnail_dir, link_path)
            print(f"[serve] Symlinked {link_path} -> {thumbnail_dir}")
        elif os.path.realpath(link_path) != thumbnail_dir:
            print(
                f"[warn] {link_path} already exists and points elsewhere. "
                f"Thumbnails may not resolve correctly.",
                file=sys.stderr,
            )

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=viz_dir,
    )
    with http.server.HTTPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"[serve] Serving {viz_dir} on {url}")
        print(f"[serve] Open your map at {url}<your_map>.html")
        print("[serve] Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] Stopped.")


def main() -> None:
    """Parse arguments and start the lightweight HTTP server."""
    p = argparse.ArgumentParser(
        description="Serve the embedding map HTML + thumbnails via HTTP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--viz-dir", required=True, help="Directory containing the HTML map file(s)."
    )
    p.add_argument("--port", type=int, default=8765, help="HTTP port to listen on.")
    p.add_argument(
        "--thumbnail-dir",
        default=None,
        help="Path to thumbnail images. If omitted, assumes "
        "they are already at <viz-dir>/thumbnails/.",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind. Use a non-loopback address only when needed.",
    )
    args = p.parse_args()
    serve(args.viz_dir, args.port, args.thumbnail_dir, host=args.host)


if __name__ == "__main__":
    main()
