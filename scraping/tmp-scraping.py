from seleniumbase import SB
import json
from datetime import datetime
import argparse
import logging
import sys
import os
from typing import List, Dict, Optional

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_SEARCH_TERM = "programador"
BASE_URL_TEMPLATE = "https://pe.computrabajo.com/trabajo-de-{}"
OUTPUT_FILE_TEMPLATE = "avisos_{}.json"
SOURCE_NAME = "computrabajo"

# Timeouts y esperas
TIMEOUT_PAGE_LOAD = 10
WAIT_AFTER_CLICK = 1.5
WAIT_BETWEEN_JOBS = 0.5
WAIT_BETWEEN_PAGES = 2
MAX_PANEL_UPDATE_ATTEMPTS = 5
PANEL_UPDATE_CHECK_INTERVAL = 0.3

# Selectores CSS
SELECTOR_CONTAINER = '#offersGridOfferContainer'
SELECTOR_ARTICLES = 'article.box_offer'
SELECTOR_JOB_LINK = 'a.js-o-link'
SELECTOR_PANEL_CONTAINER = '[data-offers-grid-detail-container]'
SELECTOR_TITLE = '[data-offers-grid-detail-title]'
SELECTOR_DESCRIPTION = '[description-offer] .fs16.t_word_wrap'
SELECTOR_NEXT_BUTTON_DISABLED = 'a.sc-LAuEU[disabled]'


class ComputrabajoScraper:
    def __init__(self, search_term: str, headless: bool = False, quiet: bool = True):
        self.search_term = search_term
        self.headless = headless
        self.quiet = quiet
        self.base_url = BASE_URL_TEMPLATE.format(search_term)
        self.jobs: List[Dict[str, str]] = []
    
    def _get_page_url(self, page_number: int) -> str:
        """Construye la URL para una página específica."""
        if page_number == 1:
            return self.base_url
        return f"{self.base_url}?p={page_number}"
    
    def _click_job_article(self, sb: SB, data_id: str) -> bool:
        """Hace clic en un artículo de trabajo usando JavaScript."""
        try:
            sb.cdp.evaluate(f'''
                document.querySelector('article[data-id="{data_id}"] a.js-o-link').click();
            ''')
            return True
        except Exception as e:
            logger.error(f"  Error al hacer clic: {e}")
            return False
    
    def _wait_for_panel_update(self, sb: SB, expected_data_id: str) -> bool:
        """Espera a que el panel se actualice con el data-id correcto."""
        for _ in range(MAX_PANEL_UPDATE_ATTEMPTS):
            try:
                panel_id = sb.cdp.evaluate(
                    f'document.querySelector("{SELECTOR_PANEL_CONTAINER}").getAttribute("data-id")'
                )
                if panel_id == expected_data_id:
                    return True
            except Exception:
                pass
            sb.sleep(PANEL_UPDATE_CHECK_INTERVAL)
        return False
    
    def _extract_title(self, sb: SB, fallback_title: str) -> str:
        """Extrae el título del panel de detalles."""
        try:
            title_element = sb.cdp.select(SELECTOR_TITLE)
            return title_element.text.strip()
        except Exception as e:
            logger.warning(f"  No se pudo extraer título: {e}")
            return fallback_title
    
    def _extract_description(self, sb: SB) -> str:
        """Extrae la descripción del trabajo."""
        try:
            description = sb.cdp.evaluate(f'''
                (() => {{
                    const elem = document.querySelector('{SELECTOR_DESCRIPTION}');
                    return elem ? elem.textContent.trim() : '';
                }})()
            ''')
            return description
        except Exception as e:
            logger.warning(f"  No se pudo extraer descripción: {e}")
            return ""
    
    def _process_job_article(self, sb: SB, article, index: int, total: int) -> Optional[Dict[str, str]]:
        """Procesa un artículo de trabajo individual."""
        try:
            data_id = article.get('data-id')
            if not data_id:
                return None
            
            enlaces = article.query_selector_all(SELECTOR_JOB_LINK)
            if not enlaces:
                return None
            
            titulo_link = enlaces[0].text.strip()
            logger.info(f"  [{index}/{total}] {titulo_link[:50]}...")
            
            # Hacer clic en el artículo
            if not self._click_job_article(sb, data_id):
                return None
            
            sb.sleep(WAIT_AFTER_CLICK)
            
            # Esperar actualización del panel
            if not self._wait_for_panel_update(sb, data_id):
                logger.warning(f"  Panel no actualizado")
                return None
            
            # Extraer información
            titulo = self._extract_title(sb, titulo_link)
            descripcion = self._extract_description(sb)
            
            return {
                'source': SOURCE_NAME,
                'scraped_at': datetime.now().isoformat(),
                'title': titulo,
                'description': descripcion
            }
            
        except Exception as e:
            logger.error(f"  Error procesando artículo: {e}")
            return None
    
    def _is_last_page(self, sb: SB) -> bool:
        """Verifica si estamos en la última página."""
        try:
            return sb.cdp.evaluate(f'''
                (() => {{
                    const btn = document.querySelector('{SELECTOR_NEXT_BUTTON_DISABLED}');
                    return btn !== null;
                }})()
            ''')
        except Exception as e:
            logger.warning(f"Error verificando última página: {e}")
            return True
    
    def _process_page(self, sb: SB, page_number: int) -> bool:
        """Procesa una página completa de resultados."""
        url = self._get_page_url(page_number)
        logger.info(f"Página {page_number}...")
        
        try:
            # Navegar
            if page_number == 1:
                sb.activate_cdp_mode(url)
            else:
                sb.cdp.open(url)
            
            sb.sleep(WAIT_BETWEEN_PAGES)
            
            # Esperar contenedor
            sb.cdp.select(SELECTOR_CONTAINER, timeout=TIMEOUT_PAGE_LOAD)
            articles = sb.cdp.select_all(f'{SELECTOR_CONTAINER} {SELECTOR_ARTICLES}')
            
            if not articles:
                logger.info("  No hay avisos en esta página")
                return False
            
            # Procesar cada artículo
            for i, article in enumerate(articles, 1):
                job_data = self._process_job_article(sb, article, i, len(articles))
                if job_data:
                    self.jobs.append(job_data)
                sb.sleep(WAIT_BETWEEN_JOBS)
            
            # Verificar si hay más páginas
            if self._is_last_page(sb):
                logger.info("  ✓ Última página alcanzada")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error en página {page_number}: {e}")
            return False
    
    def scrape(self) -> List[Dict[str, str]]:
        """Ejecuta el scraping completo."""
        logger.info(f"Buscando: {self.search_term}\n")
        
        # Redirigir stderr temporalmente si quiet=True
        old_stderr = None
        if self.quiet:
            old_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
        
        try:
            with SB(uc=True, headless=self.headless) as sb:
                page_number = 1
                while self._process_page(sb, page_number):
                    page_number += 1
        finally:
            # Restaurar stderr
            if self.quiet and old_stderr:
                sys.stderr.close()
                sys.stderr = old_stderr
        
        return self.jobs
    
    def save_to_json(self, output_file: Optional[str] = None) -> str:
        """Guarda los resultados en un archivo JSON."""
        filename = output_file or OUTPUT_FILE_TEMPLATE.format(self.search_term)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.jobs, f, ensure_ascii=False, indent=2)
        return filename


def parse_arguments() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Scraper de trabajos de Computrabajo Perú',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  %(prog)s                          # Busca "programador" (por defecto)
  %(prog)s ingeniero                # Busca "ingeniero"
  %(prog)s "data analyst"           # Busca "data analyst"
  %(prog)s vendedor --headless      # Busca "vendedor" en modo headless
  %(prog)s ingeniero -o jobs.json   # Busca y guarda en archivo específico
        '''
    )
    
    parser.add_argument(
        'search_term',
        nargs='?',
        default=DEFAULT_SEARCH_TERM,
        help=f'Término de búsqueda (default: {DEFAULT_SEARCH_TERM})'
    )
    
    parser.add_argument(
        '-o', '--output',
        metavar='FILE',
        help='Archivo de salida JSON (default: avisos_<termino>.json)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Ejecutar en modo headless (sin ventana de navegador)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Modo verbose (muestra output de Chrome y más debug)'
    )
    
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_arguments()
    
    # Configurar logging según verbosidad
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear scraper y ejecutar
    scraper = ComputrabajoScraper(
        args.search_term, 
        headless=args.headless,
        quiet=not args.verbose  # Si verbose, mostrar todo
    )
    jobs = scraper.scrape()
    
    # Guardar resultados
    filename = scraper.save_to_json(args.output)
    
    logger.info(f"\n✓ {len(jobs)} avisos guardados en {filename}")


if __name__ == "__main__":
    main()