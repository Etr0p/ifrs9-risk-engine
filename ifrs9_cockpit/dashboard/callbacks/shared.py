"""Constantes partagees entre les modules de callbacks."""

# Table styling — reutilise dans les modales et drill-downs
TABLE_HEADER_STYLE = {
    "backgroundColor": "#1E293B",
    "color": "#F8FAFC",
    "fontWeight": "bold",
}
TABLE_CELL_STYLE = {
    "backgroundColor": "#0C1222",
    "color": "#F8FAFC",
    "border": "1px solid #334155",
    "fontSize": "0.85rem",
    "padding": "8px",
}
TABLE_ODD_ROW = [{"if": {"row_index": "odd"}, "backgroundColor": "#131B2E"}]
