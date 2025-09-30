import rasterio
import os
from shapely.geometry import Polygon, mapping, Point, MultiPolygon
import time
import matplotlib.pyplot as plt
from shapely import affinity

def create_bounds_dict(path_to_tifs:str):
    files = os.listdir(path_to_tifs)
    areas_covered_by_tifs = {}
    for file in files:
        # ensure that we are handling a tif file
        if file.endswith('.tif'):
            with rasterio.open(os.path.join(path_to_tifs, file)) as dataset:
                lon_upper_left, lat_upper_left = dataset.xy(0,0)
                lon_down_right, lat_down_right = dataset.xy(dataset.height,dataset.width)
                lons_sorted = sorted([lon_upper_left,lon_down_right])
                lats_sorted = sorted([lat_upper_left, lat_down_right])
                area_covered = {'lons_sorted': lons_sorted,
                                'lats_sorted':lats_sorted
                                }
                areas_covered_by_tifs[file] =  area_covered
    return areas_covered_by_tifs

def get_path_to_tif(polygon, areas_covered_by_tifs, path_to_tif_folder): 
    # the following code has been adapted from: https://gis.stackexchange.com/a/417696, accessed: 19.04.2023, License: https://creativecommons.org/licenses/by-sa/4.0/
    mapped_poly = mapping(polygon)
    poly_ = [{"lat": coords[1], "lon":coords[0]} for coords in mapped_poly["coordinates"][0]]
    # code quotation end
    path_to_tif_file = ""
    
    for tiff in (areas_covered_by_tifs.keys()):
        tiffbounds = create_single_tiff_polygon(areas_covered_by_tifs[tiff])
        
        points_outside_tif = 0
        for point in poly_:
            #Retrieve the point
            lon, lat = point['lon'], point['lat'] 
            try:
                if not Point(lon, lat).within(tiffbounds):
                    #point from the polygon is within the given tif file
                    points_outside_tif+=1
            except:
                return ""    
        if points_outside_tif==0:
            #if true, all points from the polygon correspond to the tif file
            path_to_tif_file = os.path.join(path_to_tif_folder, tiff)
    return path_to_tif_file

def create_single_tiff_polygon(area_covered_by_single_tif: dict):
    tiff_kenya_coverage = {"minlat": None, "minlon": None, "maxlat": None, "maxlon": None}
    if (tiff_kenya_coverage["minlon"] == None):
        tiff_kenya_coverage["minlon"] = area_covered_by_single_tif["lons_sorted"][0]
    elif (area_covered_by_single_tif["lons_sorted"][0] < tiff_kenya_coverage["minlon"]):
        tiff_kenya_coverage["minlon"] = area_covered_by_single_tif["lons_sorted"][0]
        
    if (tiff_kenya_coverage["maxlon"] == None):
        tiff_kenya_coverage["maxlon"] = area_covered_by_single_tif["lons_sorted"][1]
    elif (area_covered_by_single_tif["lons_sorted"][1] > tiff_kenya_coverage["maxlon"]):
        tiff_kenya_coverage["maxlon"] = area_covered_by_single_tif["lons_sorted"][1]
        
    if (tiff_kenya_coverage["minlat"] == None):
        tiff_kenya_coverage["minlat"] = area_covered_by_single_tif["lats_sorted"][0]
    elif (area_covered_by_single_tif["lats_sorted"][0] < tiff_kenya_coverage["minlat"]):
        tiff_kenya_coverage["minlat"] = area_covered_by_single_tif["lats_sorted"][0]
        
    if (tiff_kenya_coverage["maxlat"] == None):
        tiff_kenya_coverage["maxlat"] = area_covered_by_single_tif["lats_sorted"][1]
    elif (area_covered_by_single_tif["lats_sorted"][1] > tiff_kenya_coverage["maxlat"]):
        tiff_kenya_coverage["maxlat"] = area_covered_by_single_tif["lats_sorted"][1]
    minmax_tiff = Polygon(((tiff_kenya_coverage["minlon"], tiff_kenya_coverage["minlat"]), (tiff_kenya_coverage["minlon"], tiff_kenya_coverage["maxlat"]), (tiff_kenya_coverage["maxlon"], tiff_kenya_coverage["maxlat"]), (tiff_kenya_coverage["maxlon"], tiff_kenya_coverage["minlat"]), (tiff_kenya_coverage["minlon"], tiff_kenya_coverage["minlat"])))
    minmax_tiff = minmaxpoly(minmax_tiff,0.0005)
    return minmax_tiff


def minmaxpoly(geometry, margin):
    minmaxpoly = Polygon(((min(geometry.exterior.coords.xy[0].tolist())-margin,min(geometry.exterior.coords.xy[1].tolist())-margin),
                          (min(geometry.exterior.coords.xy[0].tolist())-margin, max(geometry.exterior.coords.xy[1].tolist())+margin),
                          (max(geometry.exterior.coords.xy[0].tolist())+margin, max(geometry.exterior.coords.xy[1].tolist())+margin),
                          (max(geometry.exterior.coords.xy[0].tolist())+margin,min(geometry.exterior.coords.xy[1].tolist())-margin)
                        ))
    return minmaxpoly



def get_min_max_values_of_row_col(pixel_coordinates):
    row_values = [item[0] for item in pixel_coordinates]
    col_values = [item[1] for item in pixel_coordinates]
    
    min_row, max_row = min(row_values), max(row_values)
    min_col, max_col = min(col_values), max(col_values)
    row_len = max_row-min_row
    col_len = max_col-min_col
    if row_len > col_len:
        min_col = int(round(min_col - ((row_len-col_len)/2)))
        max_col = int(round(max_col + ((row_len-col_len)/2)))
        col_len = max_col-min_col
        
        if row_len > col_len:
            min_col = min_col -1
            col_len = max_col-min_col
        elif row_len < col_len:
            max_col = max_col -1
            col_len = max_col-min_col
    elif row_len < col_len:
        min_row = int(round(min_row - ((col_len-row_len)/2)))
        max_row = int(round(max_row + ((col_len-row_len)/2)))
        row_len = max_row-min_row
        
        if row_len < col_len:
            min_row = min_row -1
            row_len = max_row-min_row
        elif row_len > col_len:
            max_row = max_row -1
            row_len = max_row-min_row
    return {'rowminmax':[min_row, max_row], 'colminmax':[min_col, max_col]}

def save_sample(rasterio_bands_transformed, folder_to_store, image_name):
    try:
        return plt.imsave(fname=os.path.join(folder_to_store, image_name), 
               arr=rasterio_bands_transformed)
    except:
        return None
    
def create_tiff_polygon(areas_covered_by_tifs: dict):
    tiff_kenya_coverage = {"minlat": None, "minlon": None, "maxlat": None, "maxlon": None}
    for tiff in areas_covered_by_tifs.items():
        if (tiff_kenya_coverage["minlon"] == None):
            tiff_kenya_coverage["minlon"] = tiff[1]["lons_sorted"][0]
        elif (tiff[1]["lons_sorted"][0] < tiff_kenya_coverage["minlon"]):
            tiff_kenya_coverage["minlon"] = tiff[1]["lons_sorted"][0]
        
        if (tiff_kenya_coverage["maxlon"] == None):
            tiff_kenya_coverage["maxlon"] = tiff[1]["lons_sorted"][1]
        elif (tiff[1]["lons_sorted"][1] > tiff_kenya_coverage["maxlon"]):
            tiff_kenya_coverage["maxlon"] = tiff[1]["lons_sorted"][1]
        
        if (tiff_kenya_coverage["minlat"] == None):
            tiff_kenya_coverage["minlat"] = tiff[1]["lats_sorted"][0]
        elif (tiff[1]["lats_sorted"][0] < tiff_kenya_coverage["minlat"]):
            tiff_kenya_coverage["minlat"] = tiff[1]["lats_sorted"][0]
        
        if (tiff_kenya_coverage["maxlat"] == None):
            tiff_kenya_coverage["maxlat"] = tiff[1]["lats_sorted"][1]
        elif (tiff[1]["lats_sorted"][1] > tiff_kenya_coverage["maxlat"]):
            tiff_kenya_coverage["maxlat"] = tiff[1]["lats_sorted"][1]
    minmax_tiff = Polygon(((tiff_kenya_coverage["minlon"], tiff_kenya_coverage["minlat"]), (tiff_kenya_coverage["minlon"], tiff_kenya_coverage["maxlat"]), (tiff_kenya_coverage["maxlon"], tiff_kenya_coverage["maxlat"]), (tiff_kenya_coverage["maxlon"], tiff_kenya_coverage["minlat"]), (tiff_kenya_coverage["minlon"], tiff_kenya_coverage["minlat"])))
    minmax_tiff = minmaxpoly(minmax_tiff, 0.0000)
    return minmax_tiff

def get_pixel_coordinates(polygon, areas_covered_by_tifs, dataset):
    # the following code has been adapted from: https://gis.stackexchange.com/a/417696, accessed: 19.04.2023, License: https://creativecommons.org/licenses/by-sa/4.0/
    mapped_poly = mapping(polygon)
    poly_ = [{"lat": coords[1], "lon":coords[0]} for coords in mapped_poly["coordinates"][0]]
    # code quotation end
    pixel_coordinates = []
    
    for tiff in (areas_covered_by_tifs.keys()):
        tiffbounds = create_single_tiff_polygon(areas_covered_by_tifs[tiff])
        points_outside_tif = 0
        
        for point in poly_:
            #Retrieve the point
            lon, lat = point['lon'], point['lat'] 
            # security check if it is formatted properly
            try:
                if not Point(lon, lat).within(tiffbounds):
                    #point from the polygon is within the given tif file
                    points_outside_tif+=1
            except:
                break
        
        if points_outside_tif==0:
            #if true, all points from the polygon correspond to the tif file
            
            for point in poly_:
                #Retrieve the point
                lon, lat = point['lon'], point['lat']
                row_pixel, col_pixel = dataset.index(lon, lat)
                pixel_coordinates.append([row_pixel, col_pixel])
            break
    return pixel_coordinates
    
# calculate building size in square meters
def calculate_area_of_buildings(osm_data, geod):
    
    osm_data['area'] = osm_data['geometry']
    return osm_data['area'].apply(lambda x: abs(geod.geometry_area_perimeter(x)[0]))
    
    
# get all buildings that have no typeand have a building tag
def get_buildings_in_scope(osm_data):
    
    return osm_data.loc[(osm_data['type'].isnull()) & (osm_data['fclass'] == 'building')]


def calculate_centroid(polygon):
    if polygon.geom_type == "MultiPolygon":
        polygon = MultiPolygon(polygon)
    if polygon.geom_type == "Polygon":
        polygon = Polygon(polygon)
    centroid = polygon.centroid
    centroid_coordinates = f"{centroid.x:.7f}:{centroid.y:.7f}"
    return centroid_coordinates


def multipolygon_to_polygon(geometry):
    if geometry.geom_type == "MultiPolygon":
        geometry = geometry.convex_hull
    if geometry.geom_type != 'Polygon':
        return None
    return geometry


def calculate_multi_centroid(polygon_points):
    polygon = MultiPolygon(polygon_points)
    centroid = polygon.centroid
    centroid_coordinates = f"{centroid.x:.7f}:{centroid.y:.7f}"
    return centroid_coordinates


def update_db_records(osm_row):
    # Successfully update DB record with id 36.2848890:-0.7614859 and osm_id 1034597954 with image
    # Successfully update DB record with id 36.2838898:-0.7323761 and osm_id 1033782780 with image
#     time.sleep(0.2)
    db_id = calculate_centroid(osm_row.geometry)
    osm_id = osm_row.osm_id
    if db_id is None:
        print("Can not find ID in database")
        return
    try:
        document = client.get_document(
            db=DB_NAME,
            doc_id=db_id
        ).get_result()
#         if document['properties']['type_source'] == 'osm':
#             print("Record has already assigned type_source")
#             return
        document['properties']['tiff_file'] = str(osm_row.tiff_name)
        if type_source == "osm":
            document['properties']['image_ML_type'] = str(osm_row.image_ML_type)
            document['properties']['image_ML_class'] = str(osm_row.image_ML_class)
        
        document['_attachments'] = {str(db_id): {
            "content_type": "image/png",
            "data": osm_row.image_source_bytes
        }}
        # Post the batch of documents to Cloudant
        update_document_response = client.post_document(
            db=DB_NAME,
            document=document
        ).get_result()
#         print(f"Successfully update DB record with id  {db_id}  and osm_id {osm_id} with image ")
    except ApiException as ae:
        print(f"Operation  failed for {db_id} and osm_id {osm_id} and osm_id {osm_id}")
        print(" - status code: " + str(ae.code))
        print(" - error message: " + ae.message)
        if "reason" in ae.http_response.json():
            print(" - reason: " + ae.http_response.json()["reason"])


def match_corresponding_tiff(df):
    for row in tqdm(df.itertuples(), desc ="Matching corresponding tiffs"):
        df.at[row.Index, 'corresponding_tiff'] = get_path_to_tif(row.geometry, areas_covered_by_tifs, path_to_tif_folder)
    return df


def offset_polygon_coords(polygon_coords, offset_xy=(1, 1)):
    Xs, Ys = [], []
    
    for x, y in polygon_coords:
        Xs.append(x)
        Ys.append(y)
        
    x_center = min(Xs) + int((max(Xs) - min(Xs)) / 2)
    y_center = min(Ys) + int((max(Ys) - min(Ys)) / 2)
    
    offset_coords = []
    for x, y in polygon_coords:
        if (x <= x_center) and (y <= y_center):
            offset_coords.append([x - offset_xy[0], y - offset_xy[0]])
        elif (x >= x_center) and (y <= y_center):
            offset_coords.append([x + offset_xy[0], y - offset_xy[0]])
        elif (x >= x_center) and (y >= y_center):
            offset_coords.append([x + offset_xy[0], y + offset_xy[0]])
        elif (x <= x_center) and (y >= y_center):
            offset_coords.append([x - offset_xy[0], y + offset_xy[0]])
        
    return offset_coords