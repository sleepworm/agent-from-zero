# 01.1 — Define get_inventory Tool Schema and Local Data Store

> Commit: `feat: define get_inventory tool schema and local data store`
> Tag: `v0.1.1-inventory-tool-schema`

## 1. Problem

Phase 0 的 `chat()` 只能"说话"——Commit 0.1 已经亲眼见过它在没有真实数据时，要么诚实拒绝、要么（教材原本预期的）编造一个库存数字，两者都没用。要让模型第一次真正"做事"，先要把"查库存"这件事变成模型能理解的结构化描述，也要先有一份真实数据可查。这一步还不涉及模型怎么决定调用它、程序怎么执行调用结果——那是 Commit 1.2 的事，这一步只把"这个工具是什么"讲清楚。

## 2. Minimal Example

`src/tools/inventory.py`：

```python
products = {
    "A100": {"name": "Industrial Sensor A100", "price": 100, "stock": 20},
    "B200": {"name": "Industrial Sensor B200", "price": 180, "stock": 8},
}


def get_inventory(product_id: str) -> dict | None:
    return products.get(product_id)


GET_INVENTORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_inventory",
        "description": (
            "Look up the current stock level and unit price for a product, "
            "given its product ID. Use this whenever the user asks about "
            "stock, availability, or price for a specific product."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product ID, e.g. 'A100' or 'B200'.",
                },
            },
            "required": ["product_id"],
        },
    },
}
```

两个设计点：

- **`get_inventory` 找不到商品时返回 `None`，不抛异常**——这是刻意的。Commit 1.3 要验证的是"模型拿到'没找到'这个信号之后会不会诚实说明"，如果这一层直接抛异常，整条链路会在到达模型之前就被打断，反而测不到真正想测的行为。
- **`description` 是直接喂给模型的输入，不是写给人看的代码注释**——`GET_INVENTORY_SCHEMA` 里的文字会原样发给 API，模型会不会在正确的时机决定调用这个工具，很大程度上取决于这段描述写得好不好。这一点在 Commit 1.2 真正把它接到一次调用里之后才能完整验证，但写的时候就要带着这个意识——第 3 节会先做一次轻量验证。

Tool Schema 用的是 OpenAI Function Calling 协议的形状，因为 DeepSeek 走的就是这套协议——和 Commit 0.1 选 `openai` SDK 是同一个决定的延续。

### 2.0 `parameters` 那一段，其实是一份 JSON Schema

对第一次看到 Tool Schema 的读者，`parameters` 字段里这一大段 `{"type": "object", "properties": {...}, "required": [...]}` 看起来有点陌生——它不是这个项目自己发明的格式，是一个独立的、有正式规范的标准，叫 **JSON Schema**，专门用来描述"一份 JSON 数据应该长什么样"。Function Calling 协议直接借用了这套标准，来描述"调用这个函数需要传什么参数"。拆开看：

- **`"type": "object"`**——说明 `product_id` 这些参数整体上会被打包成一个 JSON 对象（也就是 Python 里的 `dict`），不是一个裸的字符串或数字。
- **`"properties"`**——列出这个对象里每一个字段分别叫什么、类型是什么。这里只有一个字段 `product_id`，类型是 `"string"`。如果一个工具需要多个参数（后面 Commit 2.1 的 `add_inventory`/`remove_inventory` 就需要两个），每个参数都是 `properties` 里的一条记录。
- **`"required"`**——列出哪些字段是必填的。这里 `product_id` 是必填的，如果模型生成的调用请求里漏了这个字段，理论上（取决于 Provider 的校验严格程度）会在真正执行前就被判定为不合法。
- **`description` 出现了两次，服务于不同的对象**——`function.description` 说明"这整个工具是干什么用的、什么时候该用它"；`properties.product_id.description` 说明"这一个参数具体是什么、长什么样"（这里给了一个例子"'A100' 或 'B200'"，帮模型判断应该往这里填什么样的字符串）。两层描述都是给模型看的，写得越具体，模型越不容易生成一个语义正确但格式不对的值（比如把商品名"Industrial Sensor A100"错填进这个本应该填 ID 的字段）。

JSON Schema 本身是一个通用规范，不止用在 Function Calling 上——很多 Web API 的请求校验、配置文件格式定义都会用它。了解这一点，是因为**后面几乎每加一个新工具，都要重新写一份这样的 `parameters`**，知道这几个关键字（`type`/`properties`/`required`）分别在描述什么，比死记硬背这一份具体的 Schema 更有用。

## 3. Break It / Observe / Why It Failed

不是破坏性实验——这一步还没有把 Schema 接到真实的模型决策链路上（那是 Commit 1.2），没有"模型会不会调用错"这类行为可以观察，真正的破坏性实验从 Commit 1.3 才开始（针对"模型会不会为不存在的商品编造库存"）。

但 DoD 要求"Tool Schema 能通过模型 API 的 schema 校验"——这一条不能只靠单元测试断言字段形状对不对，得真的发给 API 一次，看它认不认这份 Schema：

```python
r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "What tools do you have available? Just list their names and describe what each one does in one sentence."}],
    tools=[GET_INVENTORY_SCHEMA],
)
```

真实调用结果（`deepseek-chat`，2026-09-13）：

```text
schema accepted, no error
I have one tool available:

- **get_inventory** — Looks up the current stock level and unit price for a product given its product ID.
```

两件事都验证到了：API 没有因为 Schema 格式问题报错（说明 JSON Schema 的语法写对了）；模型复述出的用途和 `description` 里写的几乎一字不差（说明这段描述不只是"语法正确"，是真的被模型读懂了）。这也是为什么第 2 节要强调 `description` 是"喂给模型的输入"，不是代码注释——这次真实验证就是在确认这句话不是空话。

## 4. Implement the Fix

不适用，理由同上——这一步没有需要修复的破坏性发现。

## 5. Introduce Industrial Solution

不适用，第一次框架对比在 Phase 17。

## 6. Evaluate

```bash
pytest -v
```

实际输出：

```text
tests/test_inventory.py::test_get_inventory_returns_the_full_record_for_a_known_product PASSED
tests/test_inventory.py::test_get_inventory_returns_none_for_an_unknown_product PASSED
tests/test_inventory.py::test_products_data_matches_the_documented_scenario PASSED
tests/test_inventory.py::test_schema_has_the_shape_the_openai_compatible_api_expects PASSED
...（加上 Phase 0 的 13 个）

17 passed in 0.50s
```

四个新测试：已知商品能查到完整记录、不存在的商品返回 `None`（不是抛异常）、`products` 数据和文档描述的场景对得上、Schema 的字段形状（`type`/`function.name`/`parameters.properties`/`required`）符合 OpenAI 兼容协议的预期结构。第 3 节那次真实调用没有写进自动化测试——"这份 Schema 能不能被真实 API 接受、模型能不能读懂"是外部服务的真实行为，不是我们代码的逻辑，不能靠断言锁死一个我们控制不了的结果。

## 7. Exercises

1. 把 `description` 故意写得很含糊（比如只写 `"Get inventory."`），重新跑一次第 3 节的验证请求，看模型复述工具用途时会不会变得同样含糊——直接感受一下"描述质量"和"模型理解质量"之间的关系。
2. 给 `parameters` 加一个当前没有的可选字段（比如 `include_price: boolean`，默认不传时假设为 `true`），更新 `get_inventory` 支持它，思考"可选参数"和"必填参数"在 `required` 列表里分别怎么体现。
3. 查一下 JSON Schema 的官方规范（json-schema.org），看看除了这里用到的 `type`/`properties`/`required`，还有哪些常见关键字（比如 `enum` 限定取值范围、`minimum`/`maximum` 限定数字范围）——想一想 `product_id` 这个字段，如果想更严格地限制它只能是 `"A100"` 或 `"B200"`，应该加哪个关键字。

## 8. What We Learned

Tool Schema 的 `parameters` 字段不是这个项目自创的格式，是借用了 JSON Schema 这套通用规范来描述"调用这个函数需要传什么参数"；`description` 是直接喂给模型的输入，不是代码注释——这一点不是猜的，是真实发一次请求、看模型能不能正确复述出来验证过的；`get_inventory` 返回 `None` 而不是抛异常，是提前为 Commit 1.3 的 Hallucination 实验留的设计空间；以及验证一份 Schema"对不对"，光看代码里的字段形状不够，要真的发给 API 确认它既能被接受、也能被理解。
