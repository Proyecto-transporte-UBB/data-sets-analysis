layer_poly = QgsProject.instance().mapLayersByName("waze.routes")[0]
layer_point = QgsProject.instance().mapLayersByName("waze.routes.df")[0]
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