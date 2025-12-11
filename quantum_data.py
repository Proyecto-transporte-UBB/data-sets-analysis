import numpy as np
import pandas as pd
from datetime import datetime
from qgis.utils import iface
from qgis.core import (
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsFields,
    QgsField,
    QgsVectorLayerTemporalProperties,
    QgsDateTimeRange,
    QgsCategorizedSymbolRenderer,
    QgsMarkerSymbol,
    QgsRendererCategory
)
from PyQt5.QtCore import QVariant, QDateTime
import math

def detect_temporal_field(layer):
    if layer and isinstance(layer, QgsVectorLayer):
        temporal_properties = layer.temporalProperties()
        if temporal_properties.isActive():
            try:
                start_field = temporal_properties.startField()
                end_field = temporal_properties.endField()
                if start_field:
                    if end_field:
                        return start_field, end_field
                    else:
                        return start_field, None
            except:
                return None, None
    return None, None

def convert_to_qdatetime(timestamp):
    dt = datetime.fromtimestamp(timestamp)
    return QDateTime(dt.year, dt.month, dt.day, 
                     dt.hour, dt.minute, dt.second, 
                     int(dt.microsecond / 1000))

def quantum_data(
    layer,
    field_name,
    time_gap,
    space_gap,
    street_aligned = True,
    street_angle_tolerance = 30,
    compile_function = np.mean,
    filter_function = lambda x: True,
    batch_size = 10000
):
    total_features = layer.featureCount()
    print(f"Total features: {total_features}")
    features = []
    features0 = layer.getFeatures()
    processed_count = 0
    for feature in features0:
        if filter_function(feature):
            features.append(feature)
        processed_count += 1
        if processed_count % batch_size == 0:
            print(f"Processed {processed_count}/{total_features} features ({processed_count / total_features * 100:.1f}%)")
    print(f"Filtered features: {len(features)}")
    if not features:
        return None
    start_field, end_field = detect_temporal_field(layer)
    if field_name not in [field.name() for field in layer.fields()]:
        raise ValueError(f"Value field '{field_name}' doesn't exist in the layer")
    print("Starting quantum discretization...")
    print(f"Input layer: {layer.name()}")
    print(f"Value field: {field_name}")
    if start_field:
        print(f"Time field: {start_field}")
    n = len(features)
    times = np.zeros(n)
    lngs = np.zeros(n)
    lats = np.zeros(n)
    fields = np.zeros(n)
    for i, feature in enumerate(features):
        geom = feature.geometry()
        if geom.isMultipart():
            point = geom.asMultiPoint()[0]
        else:
            point = geom.asPoint()
        lngs[i] = point.x()
        lats[i] = point.y()
        fields[i] = feature[field_name] if feature[field_name] is not None else 0
        if start_field:
            time_val = feature[start_field]
            if isinstance(time_val, QDateTime):
                times[i] = time_val.toSecsSinceEpoch()
            elif isinstance(time_val, (datetime, pd.Timestamp)):
                times[i] = time_val.timestamp()
            elif isinstance(time_val, str):
                try:
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                        try:
                            dt = datetime.strptime(time_val, fmt)
                            times[i] = dt.timestamp()
                            break
                        except:
                            continue
                    else:
                        times[i] = i
                except:
                    times[i] = i
            else:
                times[i] = float(time_val) if time_val else i
        else:
            times[i] = i
        if i % batch_size == 0:
            print(f"Processed {i}/{n} rows ({i / n * 100}%)")
    C_lat = 111320.0
    
    def C_lng(lat_deg):
        return C_lat * math.cos(math.radians(lat_deg))
    
    if street_aligned and n >= 3:
        theta = np.zeros(n)
        for i in range(1, n-1):
            dx_deg = lngs[i+1] - lngs[i-1]
            dy_deg = lats[i+1] - lats[i-1]
            dx_m = dx_deg * C_lng(lats[i])
            dy_m = dy_deg * C_lat
            theta[i] = math.atan2(dy_m, dx_m)
        theta[0] = theta[1] if n > 1 else 0
        theta[-1] = theta[-2] if n > 1 else 0
    else:
        theta = np.zeros(n)
    x_prime = np.zeros(n)
    y_prime = np.zeros(n)
    for i in range(n):
        xG = lngs[i] * C_lng(lats[i])
        yG = lats[i] * C_lat
        cos_theta = math.cos(theta[i])
        sin_theta = math.sin(theta[i])
        x_prime[i] = cos_theta * xG - sin_theta * yG
        y_prime[i] = sin_theta * xG + cos_theta * yG
    t_min = times.min()
    t_max = times.max()
    N_t = int(math.ceil((t_max - t_min) / time_gap)) + 1
    
    def get_time_bin(t):
        bin_idx = int(math.floor((t - t_min) / time_gap))
        return max(0, min(bin_idx, N_t - 1))
    
    time_bins = np.array([get_time_bin(t) for t in times])
    x_min = x_prime.min()
    x_max = x_prime.max()
    N_x = int(math.ceil((x_max - x_min) / space_gap)) + 1
    
    def get_long_bin(x):
        bin_idx = int(math.floor((x - x_min) / space_gap))
        return max(0, min(bin_idx, N_x - 1))
    
    long_bins = np.array([get_long_bin(x) for x in x_prime])
    alpha_rad = math.radians(street_angle_tolerance)
    space_gap_y = space_gap * math.sin(alpha_rad)
    y_min = y_prime.min()
    y_max = y_prime.max()
    N_y = int(math.ceil((y_max - y_min) / space_gap_y)) + 1
    
    def get_trans_bin(y):
        bin_idx = int(math.floor((y - y_min) / space_gap_y))
        return max(0, min(bin_idx, N_y - 1))
    
    trans_bins = np.array([get_trans_bin(y) for y in y_prime])
    df = pd.DataFrame({
        'time_bin': time_bins,
        'long_bin': long_bins,
        'trans_bin': trans_bins,
        'time': times,
        'lng': lngs,
        'lat': lats,
        'field': fields,
        'theta': theta,
        'x_prime': x_prime,
        'y_prime': y_prime
    })
    grouped = df.groupby(['time_bin', 'long_bin', 'trans_bin'])
    results = []
    for (tb, lb, transb), group in grouped:
        if len(group) == 0:
            continue
        t_rep = group['time'].mean()
        lng_rep = group['lng'].mean()
        lat_rep = group['lat'].mean()
        field_rep = compile_function(group['field'].values)
        n_points = len(group)
        cos_theta_sum = group['theta'].apply(lambda x: math.cos(x)).sum()
        sin_theta_sum = group['theta'].apply(lambda x: math.sin(x)).sum()
        theta_avg = math.atan2(sin_theta_sum, cos_theta_sum)
        x_prime_avg = group['x_prime'].mean()
        y_prime_avg = group['y_prime'].mean()
        t_min_group = group['time'].min()
        t_max_group = group['time'].max()
        results.append({
            'time_bin': tb,
            'long_bin': lb,
            'trans_bin': transb,
            'time': t_rep,
            'time_start': t_min_group,
            'time_end': t_max_group,
            'lng': lng_rep,
            'lat': lat_rep,
            'field': field_rep,
            'n_points': n_points,
            'theta': theta_avg,
            'x_prime': x_prime_avg,
            'y_prime': y_prime_avg
        })
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values(['time_bin', 'x_prime', 'y_prime'])
    fields = QgsFields()
    fields.append(QgsField('id', QVariant.Int))
    fields.append(QgsField('time', QVariant.DateTime))
    fields.append(QgsField('time_start', QVariant.DateTime))
    fields.append(QgsField('time_end', QVariant.DateTime))
    fields.append(QgsField('lng', QVariant.Double))
    fields.append(QgsField('lat', QVariant.Double))
    fields.append(QgsField('field_val', QVariant.Double))
    fields.append(QgsField('n_points', QVariant.Int))
    fields.append(QgsField('theta', QVariant.Double))
    fields.append(QgsField('time_bin', QVariant.Int))
    fields.append(QgsField('long_bin', QVariant.Int))
    fields.append(QgsField('trans_bin', QVariant.Int))
    crs = layer.crs()
    layer_name = f"discretized_{field_name}_temporal"
    discretized_layer = QgsVectorLayer(f"Point?crs={crs.authid()}", layer_name, "memory")
    provider = discretized_layer.dataProvider()
    provider.addAttributes(fields)
    discretized_layer.updateFields()
    features_to_add = []
    min_time = None
    max_time = None
    for idx, row in results_df.iterrows():
        feat = QgsFeature()
        feat.setFields(fields)
        time_dt = convert_to_qdatetime(row['time'])
        time_start_dt = convert_to_qdatetime(row['time_start'])
        time_end_dt = convert_to_qdatetime(row['time_end'])
        if min_time is None or time_start_dt < min_time:
            min_time = time_start_dt
        if max_time is None or time_end_dt > max_time:
            max_time = time_end_dt
        feat.setAttribute('id', idx)
        feat.setAttribute('time', time_dt)
        feat.setAttribute('time_start', time_start_dt)
        feat.setAttribute('time_end', time_end_dt)
        feat.setAttribute('lng', float(row['lng']))
        feat.setAttribute('lat', float(row['lat']))
        feat.setAttribute('field_val', float(row['field']))
        feat.setAttribute('n_points', int(row['n_points']))
        feat.setAttribute('theta', float(row['theta']))
        feat.setAttribute('time_bin', int(row['time_bin']))
        feat.setAttribute('long_bin', int(row['long_bin']))
        feat.setAttribute('trans_bin', int(row['trans_bin']))
        point = QgsPointXY(float(row['lng']), float(row['lat']))
        feat.setGeometry(QgsGeometry.fromPointXY(point))
        features_to_add.append(feat)
    provider.addFeatures(features_to_add)
    discretized_layer.updateExtents()
    temporal_props = discretized_layer.temporalProperties()
    temporal_props.setIsActive(True)
    temporal_props.setMode(QgsVectorLayerTemporalProperties.ModeFeatureDateTimeInstantFromField)
    temporal_props.setStartField('time')
    temporal_props.setAccumulateFeatures(False)
    if min_time and max_time:
        min_time = min_time.addSecs(-time_gap)
        max_time = max_time.addSecs(time_gap)
        temporal_props.setFixedTemporalRange(QgsDateTimeRange(min_time, max_time))
    temporal_props.setFixedDuration(time_gap)
    discretized_layer.triggerRepaint()
    print("Capa temporal creada:")
    print(f"  - Nombre: {layer_name}")
    print(f"  - Características: {len(features_to_add)}")
    print(f"  - Rango temporal: {min_time.toString()} a {max_time.toString()}")
    print("  - Campo temporal: time")
    print(f"  - Intervalo temporal: {time_gap} segundos")
    return discretized_layer

def add_temporal_controller_configuration(layer):
    temporal_props = layer.temporalProperties()
    time_range = temporal_props.fixedTemporalRange()
    if time_range.isValid():
        print("\nRecomendaciones para el controlador temporal:")
        print("1. En el panel 'Controlador temporal', asegúrese de que:")
        print("   - El modo esté en 'Rango fijo'")
        print(f"   - El rango sea: {time_range.begin().toString()} a {time_range.end().toString()}")
        print(f"   - El paso de tiempo sea: {temporal_props.duration()} segundos")
        print("\n2. Para animación:")
        print("   - Use 'Reproducir' para ver la evolución temporal")
        print("   - Ajuste la velocidad en 'Intervalo de paso'")
        print("\n3. Para filtrado estático:")
        print("   - Use el deslizador para navegar en el tiempo")
        print("   - Active 'Mostrar entidades que intersectan el marco temporal'")
    return layer

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

def show_layer_fields(
    layer,
    prnt = False
):
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

def create_layer(
    input_layer_name = "",
    value_field_name = "",
    fltr = lambda x: True,
    aggregate_function = lambda x: np.mean(x),
    time_gap_seconds = 300,
    space_gap_meters = 100,
    street_aligned = True,
    street_angle_tolerance = 30,
    batch_size = 10000
):
    if len(input_layer_name) == 0:
        get_layers(True)
        return
    input_layer = [x for x in get_layers() if x.name() == input_layer_name][0]
    if len(value_field_name) == 0:
        show_layer_fields(input_layer, True)
        return
    value_field = [x for x in show_layer_fields(input_layer) if x == value_field_name][0]
    
    def filter_func(x):
        if not(fltr):
            return True
        try:
            return fltr(x)
        except:
            return False
    
    output_layer = quantum_data(
        input_layer,
        value_field,
        time_gap_seconds,
        space_gap_meters,
        street_aligned,
        street_angle_tolerance,
        aggregate_function,
        filter_func,
        batch_size
    )
    if output_layer:
        QgsProject.instance().addMapLayer(output_layer)
        QgsProject.instance().layerTreeRoot().findLayer(output_layer.id()).setItemVisibilityChecked(True)
        iface.mapCanvas().refreshAllLayers()
        iface.layerTreeView().refreshLayerSymbology(output_layer.id())
        print(f"Layer '{output_layer.name()}' added to the project")
        print(f"Verifica en TOC: {output_layer.featureCount()} features")

def copy_colors(layer_poly_name, layer_point_name):
    layer_poly = QgsProject.instance().mapLayersByName(layer_poly_name)[0]
    layer_point = QgsProject.instance().mapLayersByName(layer_point_name)[0]
    renderer_poly = layer_poly.renderer()
    renderer_point = QgsCategorizedSymbolRenderer("id", [])
    categories = []
    for cat in renderer_poly.categories():
        symbol_point = QgsMarkerSymbol.createSimple({})
        poly_symbol = cat.symbol()
        if poly_symbol:
            poly_color = poly_symbol.color()
            symbol_point.setColor(poly_color)
        new_cat = QgsRendererCategory(cat.value(), symbol_point, cat.label())
        categories.append(new_cat)
    for category in categories:
        renderer_point.addCategory(category)
    layer_point.setRenderer(renderer_point)
    layer_point.triggerRepaint()