"""Ephemeral video artwork sent as a normal image, without ARK permissions."""
from io import BytesIO
from pathlib import Path


def preview_font(size, bold=False):
    from PIL import ImageFont
    names = (['/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 'C:/Windows/Fonts/msyhbd.ttc']
             if bold else ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 'C:/Windows/Fonts/msyh.ttc'])
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    raise ValueError('video_preview_cjk_font_missing')


def fit_text(draw, text, font, width):
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text+'…', font=font) > width:
        text = text[:-1]
    return text+'…'


def render_preview(video, cover):
    from PIL import Image, ImageDraw, ImageOps
    if not cover or len(cover) > 4*1024*1024:
        raise ValueError('video_cover_size')
    with Image.open(BytesIO(cover)) as source:
        if (source.format not in {'JPEG', 'PNG', 'WEBP'} or min(source.size) < 16
                or source.width*source.height > 8_000_000):
            raise ValueError('video_cover_dimensions')
        # Preserve the complete cover; pad unusual aspect ratios instead of cropping titles.
        photo = ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'), (960, 540))
    canvas = Image.new('RGB', (960, 756), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 960, 539), fill='#edf0f4')
    canvas.paste(photo, ((960-photo.width)//2, (540-photo.height)//2))
    title_font, small = preview_font(36, True), preview_font(25)
    title = video['title']
    first = ''
    while title and draw.textlength(first+title[0], font=title_font) <= 896:
        first, title = first+title[0], title[1:]
    draw.text((32, 562), first, font=title_font, fill='#202633')
    if title:
        draw.text((32, 608), fit_text(draw, title, title_font, 896), font=title_font, fill='#202633')
    draw.line((32, 678, 928, 678), fill='#edf0f4', width=2)
    draw.text((32, 697), 'BILIBILI', font=small, fill='#e55b85')
    author = fit_text(draw, 'UP · '+(video['author'] or '未知作者'), small, 600)
    draw.text((165, 697), author, font=small, fill='#6b7280')
    minutes, seconds = divmod(video['duration_seconds'], 60)
    duration = f'{minutes:02d}:{seconds:02d}'
    width = draw.textlength(duration, font=small)
    draw.rounded_rectangle((910-width-24, 474, 932, 520), radius=10, fill='#202633')
    draw.text((910-width-12, 477), duration, font=small, fill='white')
    result = BytesIO()
    canvas.save(result, format='JPEG', quality=90, optimize=True)
    return result.getvalue()


async def fetch_cover(url):
    from .videos import COVER
    if not COVER.fullmatch(url) or not url.startswith('https://'):
        raise ValueError('video_cover_unavailable')
    import aiohttp
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12), trust_env=False,
            headers={'Referer': 'https://www.bilibili.com/'}) as session:
        async with session.get(url, allow_redirects=False) as response:
            if response.status != 200 or (response.content_length or 0) > 4*1024*1024:
                raise ValueError('video_cover_unavailable')
            data = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > 4*1024*1024:
                    raise ValueError('video_cover_size')
            return bytes(data)
