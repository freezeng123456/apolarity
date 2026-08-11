# 服务器归档与本地回收记录

归档在服务器端由正式结果目录直接生成，传输后先比较归档 SHA-256，再解包并逐项验证结果目录内的 `SHA256SUMS`。以下哈希不包含任何登录信息。

| 服务器归档 | 远端 SHA-256 | 大小（bytes） | 本地回收 SHA-256 |
|---|---|---:|---|
| `apolarity-dynamic-plate-o4-formal-v1.tar.gz` | `a850beb53b3557badead8a4becb13f5c62e235b9808853103ae4c7255dd9c36f` | 230433 | `a850beb53b3557badead8a4becb13f5c62e235b9808853103ae4c7255dd9c36f` |
| `apolarity-strain-gradient-plate-o6-formal-v1.tar.gz` | `01aca9fe0280d0a436c997937ecc058018d8733672c27cca36f4a7e5549f2f9e` | 150250 | `01aca9fe0280d0a436c997937ecc058018d8733672c27cca36f4a7e5549f2f9e` |
| `apolarity-plate-formal-smoke-v1.tar.gz` | `881546588d894d99e7cb469e1ef2f1266411f3670405359e078ddc208f326140` | 1444 | `881546588d894d99e7cb469e1ef2f1266411f3670405359e078ddc208f326140` |

远端原始结果根目录：

- `/root/apolarity-dynamic-plate-o4-formal-v1`
- `/root/apolarity-strain-gradient-plate-o6-formal-v1`

服务器端目录中的 `SHA256SUMS` 在本地解包后逐项校验通过；发布目录另有递归有限性、完成数和日志末行检查。
