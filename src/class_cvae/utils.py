
def create_img_from_text(width, height, text):
    PAD = 2
    img_size = img.shape[:2]
    text_img = (np.ones((height, width, 3)) * 255).astype(np.uint8)
    text_img = Image.fromarray(text_img)
    text_img_dr = ImageDraw.Draw(text_img)
    font = ImageFont.load_default()
    text_img_dr.text((PAD, PAD), text, font=font, fill=(0, 0, 0))
    text_img = np.array(text_img)[:, :, :1]

    return text_img