# Light 到 Ambient 的迁移

审查以已部署的 BotLife `2bac615` 为基线，保留其角色表达改动。原方案中“先注入 Ambient、再关 Light”的顺序会同时争用提示词，因此改为先仅采集验证，最终在停写窗口内一次切换两者。

1. 安装 Ambient，仅为目标物理群配置 `capture_only: true`，保持插话关闭。确认该平台入站事件可被采集；区分真实普通消息、@ 消息与合成测试。
2. 冻结写入，备份 BotLife 配置、目标 AstrBot 方案、所有相关数据库及当前插件代码。检查 SQLite 完整性、完成 WAL checkpoint。
3. 运行 `tools/migrate_light.py export`，指定源库、逻辑群、旧 bot_id 和平台/AppID/原生群。源库只读，导出目录需不存在。指定 `--config` 可额外备份配置。导出包含源文件 SHA-256、JSON 校验和、逐表内容和独立的机器人状态审阅文件。
4. 用 `import` 指向 Ambient 数据目录及备份目录。目标可有仅采集的原话，记忆必须为空。事务写入成员、明确事实、候选、摘要、当前机器人方向的互动备注；核对条数、哈希、外键和回读。重复导入同一快照不会覆盖后来的用户纠错。
5. 审阅 `configuration-suggestion.json`。在目标 AstrBot 群方案同时启用 Ambient、关闭 BotLife，关闭 Ambient 仅采集并启用所需插话。清理 BotLife 对该逻辑群的 scope、binding；只有不再被其他群使用的账号路由才可移除。移除全部退休配置字段，不将原 Light 群改成角色群。
6. 部署 BotLife 清理版本并重启，核对两个插件加载、目标配置、其他群配置、旧库哈希、迁移报告。旧源库保存为只读备份，不再被运行时打开。保留回滚材料直至用户明确批准删除。

示例（标识仅为占位）：

```sh
python tools/migrate_light.py export --source /backup/light/state.sqlite3 --destination /backup/export \
  --platform qq-instance --account APPID --group GROUP_OPENID --adapter qq_official \
  --logical-group 群名 --old-bot bot-route --config /backup/suite.json
python tools/migrate_light.py import --export /backup/export/light-export.json \
  --target-root /data/plugin_data/astrbot_plugin_ambient_dialogue --backup-dir /backup/ambient-import
```

回滚必须先停止 AstrBot，然后恢复两个插件代码、目标配置、BotLife 配置和数据库快照；保留失败迁移后的 Ambient 数据副本用于核对。不能在运行中单独恢复 Light，造成双写。迁移工具不会执行配置切换、重启或删除操作。

本地测试无法证明 QQ 官方已向应用开放普通群消息。若部署时只观察到 @ 消息，应记录这一限制，不能将“加载成功”写成旁听验收成功。
