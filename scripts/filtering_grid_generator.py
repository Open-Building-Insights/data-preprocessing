# Import necessary libraries
import shapely.wkt, shapely.geometry
import geopandas as gpd
import numpy as np
import pandas as pd
import json
import shapely



class GridGenerator:
    
    def generate_grids(self, 
                       polygon:shapely.Polygon, 
                       n:int=37
                       ) -> list[gpd.geoseries.GeoSeries, gpd.geoseries.GeoSeries]:
        assert type(polygon) == shapely.Polygon, 'argument polygon should be shapely.Polygon'
        assert type(n) == int, 'argument n should be an integer type'
        self.polygon = polygon
        self.n = n
        
        # Create a inside grid 
        inside_grid = self.rectangles_inside_polygon(self.polygon, n=self.n, tol=0)
        # Create a inside grid 
        outside_grid = self.rectangles_outside_polygon(self.polygon, n=self.n, tol=0)

        return (inside_grid, outside_grid)

        # Function to create a rectangular grid within a given polygon
    def rectangles_inside_polygon(self, polygon, n=None, size=None, tol=0, clip=True, include_poly=False) -> gpd.geoseries.GeoSeries:
        assert (n is None and size is not None) or (n is not None and size is None)
        # Extract bounding box coordinates of the polygon
        a, b, c, d = gpd.GeoSeries(polygon).total_bounds

        # Generate grids along x-axis/y-axis on the n or size
        if not n is None:
            xa = np.linspace(a, c, n + 1)
            ya = np.linspace(b, d, n + 1)
        else:
            xa = np.arange(a, c + 1, size[0])
            ya = np.arange(b, d + 1, size[1])

        # Offsets for tolerance to prevent edge cases
        if tol != 0:
            tol_xa = np.arange(0, tol * len(xa), tol)
            tol_ya = np.arange(0, tol * len(ya), tol)

        else:
            tol_xa = np.zeros(len(xa))
            tol_ya = np.zeros(len(ya))

        # Combine placements of x&y with tolerance
        xat = np.repeat(xa, 2)[1:] + np.repeat(tol_xa, 2)[:-1]
        yat = np.repeat(ya, 2)[1:] + np.repeat(tol_ya, 2)[:-1]

        # Create a grid
        grid = gpd.GeoSeries(
            [
                shapely.geometry.box(minx, miny, maxx, maxy)
                for minx, maxx in xat[:-1].reshape(len(xa) - 1, 2)
                for miny, maxy in yat[:-1].reshape(len(ya) - 1, 2)
            ]
        )

        # Ensure all returned polygons are within boundary
        if clip:
            # grid = grid.loc[grid.within(gpd.GeoSeries(np.repeat([polygon], len(grid))))]
            grid = gpd.sjoin(
                gpd.GeoDataFrame(geometry=grid),
                gpd.GeoDataFrame(geometry=[polygon]),
                how="inner",
                predicate="within",
            )["geometry"]
        # useful for visualisation
        if include_poly:
            grid = pd.concat(
                [
                    grid,
                    gpd.GeoSeries(
                        polygon.geoms
                        if isinstance(polygon, shapely.geometry.MultiPolygon)
                        else polygon
                    ),
                ]
            )
        return grid
    
    # Function to create a grid outside a given polygon
    def rectangles_outside_polygon(self, polygon, n=None, size=None, tol=0) -> gpd.geoseries.GeoSeries:
        assert (n is None and size is not None) or (n is not None and size is None)

        a, b, c, d = gpd.GeoSeries(polygon).total_bounds
        if not n is None:
            xa = np.linspace(a, c, n + 1)
            ya = np.linspace(b, d, n + 1)
        else:
            xa = np.arange(a, c + 1, size[0])
            ya = np.arange(b, d + 1, size[1])

        if tol != 0:
            tol_xa = np.arange(0, tol * len(xa), tol)
            tol_ya = np.arange(0, tol * len(ya), tol)

        else:
            tol_xa = np.zeros(len(xa))
            tol_ya = np.zeros(len(ya))

        xat = np.repeat(xa, 2)[1:] + np.repeat(tol_xa, 2)[:-1]
        yat = np.repeat(ya, 2)[1:] + np.repeat(tol_ya, 2)[:-1]

        grid = gpd.GeoSeries(
            [
                shapely.geometry.box(minx, miny, maxx, maxy)
                for minx, maxx in xat[:-1].reshape(len(xa) - 1, 2)
                for miny, maxy in yat[:-1].reshape(len(ya) - 1, 2)
            ]
        )

        # Grid for polygons outside the boundary
        result_grid = []
        # Iterate through each tile polygon in the grid
        for tile_polygon in grid:
            # Check if the tile polygon does not intersect and is not contained within the specified polygon
            if polygon.intersects(tile_polygon) == False & polygon.contains(tile_polygon) == False:
                result_grid.append(tile_polygon)
                
        return gpd.GeoSeries(result_grid)
