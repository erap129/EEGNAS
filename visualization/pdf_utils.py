import numpy
from PIL import Image as PILImage
from reportlab.platypus import Paragraph, Table, TableStyle, Image, SimpleDocTemplate
from reportlab.lib.units import cm
from reportlab.lib import utils
from reportlab.platypus.flowables import BalancedColumns
from reportlab.platypus.frames import ShowBoundaryValue


def get_image(path, width=10*cm):
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    width = min(iw, width)
    aspect = ih / float(iw)
    return Image(path, width=width, height=(width * aspect))


def create_pdf(filepath, img_paths):
    pdf = SimpleDocTemplate(filepath, rightMargin=20,
                                 leftMargin=20, topMargin=20, bottomMargin=20)
    # table = Table([[get_image(im, width=21*cm)] for im in img_paths], colWidths=600)
    # table.setStyle(TableStyle([
    #     ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
    #     ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
    #     ('BACKGROUND', (0, 0), (-1, 2), colors.lightgrey)
    # ]))
    # elements = [table]
    elements = [get_image(im) for im in img_paths]
    # story = [BalancedColumns(elements)]
    pdf.build(elements)


def create_pdf_from_story(filepath, story):
    pdf = SimpleDocTemplate(filepath, rightMargin=20,
                            leftMargin=20, topMargin=20, bottomMargin=20)
    pdf.build(story)
