const bridge = window.AstrBotPluginPage;
const el = id => document.getElementById(id);
let scope = '', offset = 0, enabled = false;
async function act(fn) {
  el('error').textContent = '';
  try { await fn(); } catch (error) { el('error').textContent = error.message; }
}
async function refresh() {
  if (!scope) { el('status').textContent = '尚未发现由 Ambient 接管的群。'; return; }
  const s = await bridge.apiGet('stickers/state', {scope, offset});
  enabled = s.enabled;
  el('enable').textContent = enabled ? '暂停本群表情' : '开启本群表情';
  el('status').textContent = `${enabled ? '已开启' : '未开启'} · 可用 ${s.ready} / ${s.target} 张\n累计有效出现 ${s.observations} 次 · 去重 ${s.unique} 张 · 待审核 ${s.pending} 张\n入库门槛 ${s.minimum} 次 · 存储 ${(s.bytes / 1048576).toFixed(2)} MiB · 下载队列 ${s.queue} · 队列溢出 ${s.queue_dropped}\n历史导入：${JSON.stringify(s.history)}\n最近采集异常：${JSON.stringify(s.last_error)}\nQQ 上传检查：${JSON.stringify(s.upload_probe)}`;
  el('items').replaceChildren();
  for (const item of s.items) {
    const card = document.createElement('article');
    if (item.preview) { const img = document.createElement('img'); img.src = 'data:image/png;base64,'+item.preview; img.alt = item.caption || '待审核图片'; card.append(img); }
    const label = document.createElement('p'); label.textContent = `${item.count} 次 · ${item.active ? '可发送' : item.state} ${item.animated ? '· 动图静态预览' : ''}`; card.append(label);
    const input = document.createElement('input'); input.value = item.caption; input.maxLength = 80; input.placeholder = '画面及适用语气'; input.setAttribute('aria-label', '表情描述'); card.append(input);
    for (const [state, title] of [['ready', '保存并通过'], ['blocked', '停用此图'], ['pending', '重新审核']]) {
      const button = document.createElement('button'); button.textContent = title;
      button.onclick = () => act(async () => { await bridge.apiPost('stickers/review', {scope, id:item.id, state, caption:input.value}); await refresh(); }); card.append(button);
    }
    el('items').append(card);
  }
  el('page').textContent = `${offset+1}—${offset+s.items.length}`;
  el('prev').disabled = offset === 0; el('next').disabled = offset+12 >= s.unique;
}
el('enable').onclick = () => act(async () => { await bridge.apiPost('stickers/enable', {scope, enabled:!enabled}); await refresh(); });
el('import').onclick = () => act(async () => { await bridge.apiPost('stickers/import', {scope}); await refresh(); });
el('probe').onclick = () => act(async () => { await bridge.apiPost('stickers/probe', {scope}); await refresh(); });
el('refresh').onclick = () => act(refresh);
el('group').onchange = () => act(async () => { scope=el('group').value; offset=0; await refresh(); });
el('prev').onclick = () => act(async () => { offset=Math.max(0,offset-12); await refresh(); });
el('next').onclick = () => act(async () => { offset+=12; await refresh(); });
await act(async () => {
  await bridge.ready();
  const result = await bridge.apiGet('stickers/state');
  for (const group of result.groups) { const option = document.createElement('option'); option.value=group.scope; option.textContent=group.name; el('group').append(option); }
  scope=el('group').value; await refresh();
});
