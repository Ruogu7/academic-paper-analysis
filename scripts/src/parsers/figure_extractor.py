"""Figure extractor for academic papers."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class ExtractedFigure:
    """Represents an extracted figure."""

    figure_id: str
    image: Optional[Image.Image] = None
    image_bytes: Optional[bytes] = None
    page_number: int = 0
    bbox: Optional[tuple[int, int, int, int]] = None
    caption: str = ""


class FigureExtractor:
    """Extract figures from PDF files."""

    def __init__(self):
        if fitz is None:
            raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
        if Image is None:
            raise ImportError("Pillow is required. Install with: pip install pillow")

    def extract_figures_from_pdf(
        self,
        pdf_path: str | Path,
        output_dir: Optional[Path] = None,
        dpi: int = 150,
    ) -> list[ExtractedFigure]:
        """Extract all figures from a PDF."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        extracted_figures = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get images from page
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Create PIL Image
                    image = Image.open(io.BytesIO(image_bytes))

                    figure = ExtractedFigure(
                        figure_id=f"page{page_num + 1}_img{img_index + 1}",
                        image=image,
                        image_bytes=image_bytes,
                        page_number=page_num + 1,
                    )
                    extracted_figures.append(figure)

                    # Save to file if output_dir specified
                    if output_dir:
                        self._save_image(image, output_dir, figure.figure_id)

                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")

        doc.close()
        logger.info(f"Extracted {len(extracted_figures)} figures from {pdf_path.name}")
        return extracted_figures

    def extract_figure_by_bbox(
        self,
        pdf_path: str | Path,
        page_number: int,
        bbox: tuple[int, int, int, int],
        output_path: Optional[Path] = None,
    ) -> Optional[ExtractedFigure]:
        """Extract a specific region as a figure."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))

        try:
            page = doc[page_number - 1]  # 0-indexed

            # Render the region to an image
            zoom = 2.0  # Higher quality
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=bbox)

            # Convert to PIL Image
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            figure = ExtractedFigure(
                figure_id=f"page{page_number}_region",
                image=image,
                image_bytes=img_bytes,
                page_number=page_number,
                bbox=bbox,
            )

            if output_path:
                self._save_image(image, output_path.parent, output_path.stem)

            return figure

        except Exception as e:
            logger.error(f"Failed to extract figure: {e}")
            return None
        finally:
            doc.close()

    def extract_page_as_image(
        self,
        pdf_path: str | Path,
        page_number: int,
        dpi: int = 150,
    ) -> Optional[Image.Image]:
        """Extract an entire page as an image."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))

        try:
            page = doc[page_number - 1]
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            img_bytes = pix.tobytes("png")
            return Image.open(io.BytesIO(img_bytes))

        except Exception as e:
            logger.error(f"Failed to extract page as image: {e}")
            return None
        finally:
            doc.close()

    def _save_image(self, image: Image.Image, output_dir: Path, filename: str) -> Path:
        """Save image to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{filename}.png"
        image.save(output_path, "PNG")
        logger.debug(f"Saved figure to {output_path}")
        return output_path
