regions = {
    "cont_shelf":  ([-75, -72], [248, 258]),
    "pig":         ([-76, -74], [256.5, 263]),
    "thwaites":    ([-75.7, -74.7], [252.2, 255.3]),
    "abbot":       ([-73.3, -72], [256, 271.8]),
    "dotson_crosson":      ([-75.2, -74], [245.5, 251]),
    "getz":  ([-75, -73], [224, 245.2]),
}

transport_lat, transport_lon = -73, [251, 254.5]

ATM_PRESS = 101325      # Atmospheric pressure in Pascals
RHO_REF = 1025          # Density of seawater in kg/m³
G = 9.81                # Gravity
SV = 1e-6               # Sverdrups
R_EARTH = 6371.0        # Earth's radius

depth_range = (-200,-700)
depth_limit = 1500