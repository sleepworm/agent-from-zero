from src.tools.inventory import GET_INVENTORY_SCHEMA, get_inventory, products


def test_get_inventory_returns_the_full_record_for_a_known_product():
    result = get_inventory("A100")

    assert result == {"name": "Industrial Sensor A100", "price": 100, "stock": 20}


def test_get_inventory_returns_none_for_an_unknown_product():
    result = get_inventory("Z999")

    assert result is None


def test_products_data_matches_the_documented_scenario():
    assert products["A100"]["stock"] == 20
    assert products["B200"]["stock"] == 8


def test_schema_has_the_shape_the_openai_compatible_api_expects():
    assert GET_INVENTORY_SCHEMA["type"] == "function"

    function = GET_INVENTORY_SCHEMA["function"]
    assert function["name"] == "get_inventory"
    assert isinstance(function["description"], str) and len(function["description"]) > 0

    params = function["parameters"]
    assert params["type"] == "object"
    assert "product_id" in params["properties"]
    assert params["properties"]["product_id"]["type"] == "string"
    assert params["required"] == ["product_id"]
