import numpy as np
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsField, QgsFields, QgsVectorFileWriter, QgsCoordinateReferenceSystem, QgsProcessingFeedback, QgsFeatureSink
from qgis.PyQt.QtCore import QVariant, QDateTime
from datetime import datetime, timedelta
import math

METERS_PER_DEGREE_LAT = 111320

def meters_per_degree_lng(lat_deg):
    return METERS_PER_DEGREE_LAT * math.cos(math.radians(lat_deg))

def calculate_street_headings(lng_data, lat_data):
    n = len(lng_data)
    headings = np.zeros(n)
    for i in range(1, n-1):
        dx_meters = (lng_data[i+1] - lng_data[i-1]) * meters_per_degree_lng(lat_data[i])
        dy_meters = (lat_data[i+1] - lat_data[i-1]) * METERS_PER_DEGREE_LAT
        headings[i] = math.atan2(dy_meters, dx_meters)
    if n > 1:
        headings[0] = headings[1]
        headings[-1] = headings[-2]
    return headings

def transform_to_street_coords(lng_data, lat_data, headings):
    n = len(lng_data)
    x_along = np.zeros(n)
    y_across = np.zeros(n)
    for i in range(n):
        lat_meters = lat_data[i] * METERS_PER_DEGREE_LAT
        lng_meters = lng_data[i] * meters_per_degree_lng(lat_data[i])
        angle = headings[i]
        cos_ang = math.cos(angle)
        sin_ang = math.sin(angle)
        x_along[i] = lng_meters * cos_ang + lat_meters * sin_ang
        y_across[i] = -lng_meters * sin_ang + lat_meters * cos_ang
    return x_along, y_across

def detect_temporal_field(layer):
    if not hasattr(layer, 'isTemporal'):
        return None
    if not layer.isTemporal():
        return None
    temporal_props = layer.temporalProperties()
    if temporal_props.isActive():
        mode = temporal_props.mode()
        if mode == 1:
            field = temporal_props.startField()
            return field
        elif mode == 2:
            field = temporal_props.startField()
            return field
        elif mode == 3:
            expression = temporal_props.startExpression()
            import re
            match = re.search(r'\"([^\"]+)\"', expression)
            if match:
                return match.group(1)
    return None

def process_features_in_batches(layer, time_field, value_field, batch_size=10000):
    times = []
    lngs = []
    lats = []
    values = []
    total_features = layer.featureCount()
    print(f"Total features to process: {total_features:,}")
    for start_idx in range(0, total_features, batch_size):
        end_idx = min(start_idx + batch_size, total_features)
        print(f"Processing features {start_idx:,} to {end_idx:,}...")
        features = []
        for i, feature in enumerate(layer.getFeatures()):
            if i >= start_idx and i < end_idx:
                features.append(feature)
            elif i >= end_idx:
                break
        for feature in features:
            geom = feature.geometry()
            if geom.isNull() or not geom.type() == 0:
                continue
            point = geom.asPoint()
            lngs.append(point.x())
            lats.append(point.y())
            if time_field:
                time_val = feature[time_field]
                if isinstance(time_val, (datetime, QDateTime)):
                    if isinstance(time_val, QDateTime):
                        time_val = time_val.toPyDateTime()
                    times.append(time_val.timestamp())
                elif isinstance(time_val, (int, float)):
                    times.append(time_val)
                else:
                    try:
                        time_str = str(time_val)
                        try:
                            dt = datetime.fromisoformat(time_str)
                        except ValueError:
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', 
                                       '%d/%m/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                                try:
                                    dt = datetime.strptime(time_str, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                raise ValueError(f"No se pudo parsear el tiempo: {time_str}")
                        times.append(dt.timestamp())
                    except Exception as e:
                        print(f"Warning: Could not parse time value '{time_val}': {e}")
                        times.append(0)
            else:
                times.append(0)
            val = feature[value_field]
            if isinstance(val, QVariant):
                val = val.value()
            values.append(float(val) if val is not None else 0.0)
        del features
    return np.array(times), np.array(lngs), np.array(lats), np.array(values)

def quantum_discretization_qgis(input_layer, value_field, time_gap_seconds = 5, space_gap_meters = 50, street_aligned = True, street_angle_tolerance = 30, aggregate_function = lambda x: np.mean(x)):
    time_field = detect_temporal_field(input_layer)
    if value_field not in [field.name() for field in input_layer.fields()]:
        raise ValueError(f"Value field '{value_field}' doesn't exist in the layer")
    print(f"Starting quantum discretization...")
    print(f"Input layer: {input_layer.name()}")
    print(f"Value field: {value_field}")
    print(f"Time field: {time_field if time_field else 'Not detected'}")
    times, lngs, lats, values = process_features_in_batches(input_layer, time_field, value_field)
    n = len(times)
    if n == 0:
        raise ValueError("No valid point features found in the layer")
    print(f"Processed {n:,} valid point features")
    has_time_field = bool(time_field)
    has_varied_times = len(np.unique(times)) > 1 if len(times) > 0 else False
    if not has_time_field or not has_varied_times:
        print("No valid temporal field detected or times are constant. Performing spatial-only discretization.")
        time_bin_indices = np.zeros(len(times), dtype=int)
    else:
        print(f"Temporal discretization enabled with time gap: {time_gap_seconds} seconds")
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        lngs = lngs[sort_idx]
        lats = lats[sort_idx]
        values = values[sort_idx]
        min_time = np.min(times)
        max_time = np.max(times)
        time_bins = np.arange(min_time, max_time + time_gap_seconds, time_gap_seconds)
        if len(time_bins) == 0:
            time_bins = np.array([min_time, max_time])
        time_bin_indices = np.digitize(times, time_bins) - 1
        time_bin_indices = np.clip(time_bin_indices, 0, len(time_bins) - 2)
    if street_aligned:
        print(f"Spatial discretization: Street-aligned with gap: {space_gap_meters} meters")
        headings = calculate_street_headings(lngs, lats)
        x_along, y_across = transform_to_street_coords(lngs, lats, headings)
        min_x_along = np.min(x_along)
        max_x_along = np.max(x_along)
        x_bins = np.arange(min_x_along, max_x_along + space_gap_meters, space_gap_meters)
        if len(x_bins) == 0:
            x_bins = np.array([min_x_along, max_x_along])
        x_bin_indices = np.digitize(x_along, x_bins) - 1
        x_bin_indices = np.clip(x_bin_indices, 0, len(x_bins) - 2)
        cross_gap = space_gap_meters * math.sin(math.radians(street_angle_tolerance))
        min_y_across = np.min(y_across)
        max_y_across = np.max(y_across)
        y_bins = np.arange(min_y_across, max_y_across + cross_gap, cross_gap)
        if len(y_bins) == 0:
            y_bins = np.array([min_y_across, max_y_across])
        y_bin_indices = np.digitize(y_across, y_bins) - 1
        y_bin_indices = np.clip(y_bin_indices, 0, len(y_bins) - 2)
        
        if has_time_field and has_varied_times:
            bin_combinations = np.column_stack([time_bin_indices, x_bin_indices, y_bin_indices])
        else:
            bin_combinations = np.column_stack([x_bin_indices, y_bin_indices])
    else:
        print(f"Spatial discretization: Grid-aligned with gap: {space_gap_meters} meters")
        min_lat = np.min(lats)
        max_lat = np.max(lats)
        min_lng = np.min(lngs)
        max_lng = np.max(lngs)
        mean_lat = np.mean(lats)
        lat_meters_range = (max_lat - min_lat) * METERS_PER_DEGREE_LAT
        lng_meters_range = (max_lng - min_lng) * meters_per_degree_lng(mean_lat)
        lat_bins_meters = np.arange(0, lat_meters_range + space_gap_meters, space_gap_meters)
        lng_bins_meters = np.arange(0, lng_meters_range + space_gap_meters, space_gap_meters)
        lat_bins = min_lat + lat_bins_meters / METERS_PER_DEGREE_LAT
        lng_bins = min_lng + lng_bins_meters / meters_per_degree_lng(mean_lat)
        if len(lat_bins) == 0:
            lat_bins = np.array([min_lat, max_lat])
        if len(lng_bins) == 0:
            lng_bins = np.array([min_lng, max_lng])
        lat_bin_indices = np.digitize(lats, lat_bins) - 1
        lat_bin_indices = np.clip(lat_bin_indices, 0, len(lat_bins) - 2)
        lng_bin_indices = np.digitize(lngs, lng_bins) - 1
        lng_bin_indices = np.clip(lng_bin_indices, 0, len(lng_bins) - 2)
        if has_time_field and has_varied_times:
            bin_combinations = np.column_stack([time_bin_indices, lat_bin_indices, lng_bin_indices])
        else:
            bin_combinations = np.column_stack([lat_bin_indices, lng_bin_indices])
    print(f"Calculating unique combinations...")
    unique_combinations, inverse_indices = np.unique(bin_combinations, axis=0, return_inverse=True)
    print(f"Found {len(unique_combinations):,} unique combinations")
    result_times = []
    result_lngs = []
    result_lats = []
    result_values = []
    result_counts = []
    result_headings = [] if street_aligned else None
    print("Aggregating results...")
    for i, combo in enumerate(unique_combinations):
        mask = inverse_indices == i
        count = np.sum(mask)
        if count == 0:
            continue
        group_times = times[mask]
        group_lngs = lngs[mask]
        group_lats = lats[mask]
        group_values = values[mask]
        if has_time_field and has_varied_times:
            avg_time = np.mean(group_times)
            result_times.append(datetime.fromtimestamp(avg_time))
        else:
            result_times.append(datetime.now())
        avg_lng = np.mean(group_lngs)
        avg_lat = np.mean(group_lats)
        result_lngs.append(avg_lng)
        result_lats.append(avg_lat)
        try:
            agg_value = aggregate_function(group_values)
        except Exception as e:
            raise ValueError(f"Error applying aggregate function: {str(e)}")
        result_values.append(agg_value)
        result_counts.append(count)
        if street_aligned:
            group_headings = headings[mask]
            cos_mean = np.mean(np.cos(group_headings))
            sin_mean = np.mean(np.sin(group_headings))
            avg_heading = math.atan2(sin_mean, cos_mean)
            result_headings.append(math.degrees(avg_heading))
        if i % 1000 == 0:
            print(f"  Processed {i:,}/{len(unique_combinations):,} combinations...")
    print(f"Creating output layer with {len(result_times):,} features...")
    fields = QgsFields()
    fields.append(QgsField("id", QVariant.Int))
    if has_time_field and has_varied_times:
        fields.append(QgsField("time", QVariant.DateTime))
    fields.append(QgsField("value", QVariant.Double))
    fields.append(QgsField("n_points", QVariant.Int))
    if street_aligned:
        fields.append(QgsField("heading", QVariant.Double))
        if has_time_field and has_varied_times:
            fields.append(QgsField("time_gap", QVariant.Int))
        fields.append(QgsField("space_gap", QVariant.Int))
    crs = input_layer.crs()
    if not crs.isValid():
        crs = QgsCoordinateReferenceSystem("EPSG:4326")
    output_layer = QgsVectorLayer("Point?crs=" + crs.authid(), f"quantum_discretized_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "memory")
    output_layer.dataProvider().addAttributes(fields)
    output_layer.updateFields()
    batch_size_features = 1000
    output_features = []
    for i in range(len(result_times)):
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(result_lngs[i], result_lats[i])))
        attrs = [i + 1]
        if has_time_field and has_varied_times:
            attrs.append(result_times[i])
        attrs.extend([result_values[i], result_counts[i]])
        if street_aligned:
            attrs.append(result_headings[i])
            if has_time_field and has_varied_times:
                attrs.append(time_gap_seconds)
            attrs.append(space_gap_meters)
        feat.setAttributes(attrs)
        output_features.append(feat)
        if len(output_features) >= batch_size_features:
            output_layer.dataProvider().addFeatures(output_features)
            output_features = []
    if output_features:
        output_layer.dataProvider().addFeatures(output_features)
    output_layer.updateExtents()
    print(f"Quantum discretization completed successfully!")
    print(f"Input features: {n:,}")
    print(f"Output features: {len(result_times):,}")
    print(f"Compression ratio: {n/len(result_times):.1f}:1")
    return output_layer

def get_layers(prnt = False):
    layers = QgsProject.instance().mapLayers()
    if not layers:
        if prnt: print("There's no layers in project.")
        return None
    if prnt:
      print("=" * 60)
      print(f"AVIABLE LAYERS ({len(layers)}):")
      print("=" * 60)
    for i, (layer_id, layer) in enumerate(layers.items(), 1):
        geom_info = ""
        if isinstance(layer, QgsVectorLayer):
            geom_type = layer.geometryType()
            if geom_type == 0:
                geom_info = "(Points)"
            elif geom_type == 1:
                geom_info = "(Lines)"
            elif geom_type == 2:
                geom_info = "(Polygons)"
        if prnt: print(f"{i:3d}. {layer.name():30} {geom_info:15} | Features: {layer.featureCount() if hasattr(layer, 'featureCount') else 'N/A'}")
    return list(layers.values())

def show_layer_fields(layer, prnt = False):
    if not layer or not isinstance(layer, QgsVectorLayer):
        if prnt: print("Invalid or not vectorial layer.")
        return []
    fields = layer.fields()
    field_names = [field.name() for field in fields]
    if prnt:
      print(f"\Aviable fields in '{layer.name()}':")
      print("-" * 40)
      for i, field in enumerate(fields):
          print(f"{i + 1:3d}. {field.name():20} | Type: {field.typeName():15}")
    return field_names

def excecute_functions(n, m):
  if n < 1:
    get_layers(True)
    return
  input_layer = get_layers()[n - 1]
  if m < 1:
    show_layer_fields(input_layer, True)
    return
  value_field = show_layer_fields(input_layer)[m - 1]
  return quantum_discretization_qgis(input_layer, value_field)

excecute_functions(3, 14)
