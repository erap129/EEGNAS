from reportlab.platypus import Image, SimpleDocTemplate
from reportlab.lib.units import cm
from reportlab.lib import utils


def get_image(path, width=18*cm):
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    width = min(iw, width)
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))


def create_pdf(filepath, img_paths):
    pdf = SimpleDocTemplate(filepath, rightMargin=20,
                                 leftMargin=20, topMargin=20, bottomMargin=20)
    elements = [get_image(im, width=10*cm) for im in img_paths]
    pdf.build(elements)


def create_pdf_from_story(filepath, story):
    pdf = SimpleDocTemplate(filepath, rightMargin=3,
                            leftMargin=3, topMargin=3, bottomMargin=3)
    pdf.build(story)
