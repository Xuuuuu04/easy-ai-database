# 运行与排错

## 启动检查
- 后端健康检查：`GET /health`
- 前端访问：`http://localhost:5173`

## 常见问题
1. **模型不可用**
   - 检查 `.env` 的 `LLM_BASE_URL`/`EMBED_BASE_URL`
   - 确认本地模型服务已启动

2. **无引用拒答**
   - 默认需要至少 1 条引用；可通过 `REQUIRE_CITATIONS=0` 关闭

3. **索引异常**
   - 删除 `./data/index` 重新导入

## 测试模式
- `MOCK_MODE=1`：跳过向量与模型调用，使用简单关键词检索
