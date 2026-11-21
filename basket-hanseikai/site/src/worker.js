// Family basketball review app: static page + small API. Everything lives in one R2 bucket:
//   config/auth.json   family passphrase (salted hash) and the session token
//   db/<col>.json      videos / scenes / boards, each a map of id -> document
//   blobs/<id>         game videos, clips and still images
import PAGE from '../public/index.html';

const COLS = ['videos', 'scenes', 'boards'];
const MAX_BYTES = 10 * 1024 * 1024 * 1024; // R2 free allowance, shown in the storage meter
const TYPES = /^(image\/jpeg|image\/png|video\/mp4|video\/webm|video\/quicktime)$/;
const BLOB_ID = /^[a-f0-9]{48}$/;
const DOC_ID = /^[A-Za-z0-9_-]{1,40}$/;

const json = (o, status = 200, headers = {}) =>
  new Response(JSON.stringify(o), { status, headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers } });
const hex = (buf) => [...new Uint8Array(buf)].map((x) => x.toString(16).padStart(2, '0')).join('');
const sha = async (s) => hex(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s)));
const rnd = () => hex(crypto.getRandomValues(new Uint8Array(24)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const isObj = (v) => v && typeof v === 'object' && !Array.isArray(v);
function merge(a, b) {
  const o = { ...a };
  for (const k in b) o[k] = isObj(b[k]) && isObj(a && a[k]) ? merge(a[k], b[k]) : b[k];
  return o;
}
const cookieToken = (req) => ((req.headers.get('cookie') || '').match(/(?:^|;\s*)hk=([a-f0-9]+)/) || [])[1] || '';
const sessionCookie = (t) => `hk=${t}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Lax`;

export default {
  async fetch(req, env) {
    const url = new URL(req.url);
    if (!url.pathname.startsWith('/api/') && !url.pathname.startsWith('/media/')) {
      if (url.pathname !== '/') return new Response('Not found', { status: 404 });
      return new Response(PAGE, { headers: { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-cache', 'x-robots-tag': 'noindex' } });
    }
    try {
      return await handle(req, env, url);
    } catch (e) {
      return json({ error: 'server', message: String((e && e.message) || e) }, 500);
    }
  },
};

async function handle(req, env, url) {
  const p = url.pathname, m = req.method, q = url.searchParams;
  const cfgObj = await env.B.get('config/auth.json');
  const cfg = cfgObj ? await cfgObj.json() : null;
  const authed = !!cfg && cookieToken(req) === cfg.token;

  if (p === '/api/me') return json({ setup: !!cfg, authed });
  if (p === '/api/setup' && m === 'POST') {
    if (cfg) return json({ error: 'already' }, 409);
    const pass = String((await req.json()).pass || '');
    if (pass.length < 4) return json({ error: 'short' }, 400);
    const salt = rnd(), token = rnd();
    await env.B.put('config/auth.json', JSON.stringify({ salt, hash: await sha(salt + pass), token }));
    return json({ ok: true }, 200, { 'set-cookie': sessionCookie(token) });
  }
  if (p === '/api/login' && m === 'POST') {
    if (!cfg) return json({ error: 'nosetup' }, 409);
    const pass = String((await req.json()).pass || '');
    if ((await sha(cfg.salt + pass)) !== cfg.hash) { await sleep(800); return json({ error: 'wrong' }, 403); }
    return json({ ok: true }, 200, { 'set-cookie': sessionCookie(cfg.token) });
  }
  if (!authed) return json({ error: 'auth' }, 401);

  if (p === '/api/state' && m === 'GET') {
    const objs = await Promise.all(COLS.map((c) => env.B.get(`db/${c}.json`)));
    const v = objs.map((o) => (o ? o.etag : '-')).join('.');
    if (q.get('v') === v) return json({ same: true, v });
    const out = { v };
    for (let i = 0; i < COLS.length; i++) out[COLS[i]] = objs[i] ? await objs[i].json() : {};
    return json(out);
  }
  if (p === '/api/doc' && m === 'POST') {
    const { col, id, op, data } = await req.json();
    if (!COLS.includes(col) || !DOC_ID.test(String(id)) || !['set', 'update', 'del'].includes(op)) return json({ error: 'bad' }, 400);
    if (op !== 'del' && !isObj(data)) return json({ error: 'bad' }, 400);
    const key = `db/${col}.json`;
    for (let attempt = 0; attempt < 6; attempt++) {
      const cur = await env.B.get(key);
      const map = cur ? await cur.json() : {};
      if (op === 'set') map[id] = data;
      else if (op === 'del') delete map[id];
      else { if (!map[id]) return json({ error: 'missing' }, 404); map[id] = merge(map[id], data); }
      const done = await env.B.put(key, JSON.stringify(map), cur ? { onlyIf: { etagMatches: cur.etag } } : undefined);
      if (done) return json({ ok: true });
      await sleep(40 * (attempt + 1)); // someone else wrote in between: read again and reapply
    }
    return json({ error: 'busy' }, 503);
  }
  if (p === '/api/blob' && m === 'PUT') {
    const type = q.get('type') || '';
    if (!TYPES.test(type)) return json({ error: 'unsupported_type' }, 400);
    if (+req.headers.get('content-length') > 95 * 1024 * 1024) return json({ error: 'too_large' }, 413);
    const id = rnd();
    await env.B.put('blobs/' + id, req.body, { httpMetadata: { contentType: type } });
    return json({ id });
  }
  if (p === '/api/blob' && m === 'DELETE') {
    const id = q.get('id') || '';
    if (!BLOB_ID.test(id)) return json({ error: 'bad' }, 400);
    await env.B.delete('blobs/' + id);
    return json({ ok: true });
  }
  if (p === '/api/mp/create' && m === 'POST') {
    const type = q.get('type') || '';
    if (!TYPES.test(type)) return json({ error: 'unsupported_type' }, 400);
    const id = rnd();
    const mp = await env.B.createMultipartUpload('blobs/' + id, { httpMetadata: { contentType: type } });
    return json({ id, uploadId: mp.uploadId });
  }
  if (p === '/api/mp/part' && m === 'PUT') {
    const id = q.get('id') || '', n = +q.get('n');
    if (!BLOB_ID.test(id) || !(n >= 1 && n <= 10000)) return json({ error: 'bad' }, 400);
    const part = await env.B.resumeMultipartUpload('blobs/' + id, q.get('uploadId') || '').uploadPart(n, req.body);
    return json({ n: part.partNumber, etag: part.etag });
  }
  if (p === '/api/mp/complete' && m === 'POST') {
    const { id, uploadId, parts } = await req.json();
    if (!BLOB_ID.test(String(id)) || !Array.isArray(parts)) return json({ error: 'bad' }, 400);
    await env.B.resumeMultipartUpload('blobs/' + id, String(uploadId)).complete(parts);
    return json({ id });
  }
  if (p === '/api/mp/abort' && m === 'POST') {
    const { id, uploadId } = await req.json();
    if (!BLOB_ID.test(String(id))) return json({ error: 'bad' }, 400);
    try { await env.B.resumeMultipartUpload('blobs/' + id, String(uploadId)).abort(); } catch (e) {}
    return json({ ok: true });
  }
  if (p === '/api/usage' && m === 'GET') {
    let bytes = 0, files = 0, cursor;
    for (let i = 0; i < 10; i++) {
      const l = await env.B.list({ prefix: 'blobs/', cursor });
      for (const o of l.objects) { bytes += o.size; files++; }
      if (!l.truncated) break;
      cursor = l.cursor;
    }
    return json({ bytes, files, maxBytes: MAX_BYTES });
  }
  if (p.startsWith('/media/') && (m === 'GET' || m === 'HEAD')) {
    const id = p.slice(7);
    if (!BLOB_ID.test(id)) return json({ error: 'bad' }, 400);
    const hasRange = req.headers.has('range');
    const obj = m === 'HEAD' ? await env.B.head('blobs/' + id) : await env.B.get('blobs/' + id, { range: req.headers, onlyIf: req.headers });
    if (!obj) return json({ error: 'missing' }, 404);
    const h = new Headers();
    obj.writeHttpMetadata(h);
    h.set('etag', obj.httpEtag);
    h.set('accept-ranges', 'bytes');
    h.set('cache-control', 'private, max-age=86400');
    if (m === 'HEAD') { h.set('content-length', obj.size); return new Response(null, { headers: h }); }
    if (!obj.body) return new Response(null, { status: 304, headers: h });
    if (hasRange && obj.range) {
      const offset = obj.range.offset ?? Math.max(0, obj.size - (obj.range.suffix || 0));
      const length = obj.range.length ?? obj.size - offset;
      h.set('content-range', `bytes ${offset}-${offset + length - 1}/${obj.size}`);
      h.set('content-length', length);
      return new Response(obj.body, { status: 206, headers: h });
    }
    h.set('content-length', obj.size);
    return new Response(obj.body, { headers: h });
  }
  return json({ error: 'notfound' }, 404);
}
