"""
Enhanced Playwright scraper specifically optimized for e-commerce product pages.

This scraper intelligently extracts product information while filtering out UI noise.
"""

import asyncio
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from ...utils.logging_config import get_logger

logger = get_logger(__name__)


class ProductContentExtractor:
    """Intelligent product content extraction from e-commerce pages"""

    @staticmethod
    def extract_product_cards(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract individual product cards from listing pages"""
        products = []

        # Common product card selectors
        product_selectors = [
            {"class_": re.compile(r"product[-_]?(card|item|tile|grid[-_]?item)", re.I)},
            {"class_": re.compile(r"collection[-_]?item", re.I)},
            {"data-product-id": True},
            {"itemtype": re.compile(r"schema.org/Product", re.I)},
        ]

        for selector in product_selectors:
            product_elements = soup.find_all(["div", "article", "li"], **selector)

            for element in product_elements:
                product_data = {
                    "title": "",
                    "description": "",
                    "price": "",
                    "link": "",
                    "image": "",
                    "availability": "",
                    "sku": "",
                }

                # Extract product title
                title_elem = element.find(
                    ["h2", "h3", "h4", "a"],
                    class_=re.compile(r"title|name|product[-_]?name", re.I),
                )
                if title_elem:
                    product_data["title"] = title_elem.get_text(strip=True)

                # Extract product link
                link_elem = element.find("a", href=True)
                if link_elem:
                    product_data["link"] = link_elem.get("href", "")

                # Extract price
                price_elem = element.find(
                    ["span", "div", "p"], class_=re.compile(r"price", re.I)
                )
                if price_elem:
                    product_data["price"] = price_elem.get_text(strip=True)

                # Extract description
                desc_elem = element.find(
                    ["p", "div"],
                    class_=re.compile(r"description|excerpt|summary", re.I),
                )
                if desc_elem:
                    product_data["description"] = desc_elem.get_text(strip=True)

                # Extract SKU if available
                sku_elem = element.find(["span", "div"], attrs={"data-sku": True})
                if sku_elem:
                    product_data["sku"] = sku_elem.get("data-sku", "")

                # Extract availability
                avail_elem = element.find(
                    ["span", "div"], class_=re.compile(r"stock|availability", re.I)
                )
                if avail_elem:
                    product_data["availability"] = avail_elem.get_text(strip=True)

                # Only add if we have at least a title
                if product_data["title"]:
                    products.append(product_data)

        return products

    @staticmethod
    def extract_single_product_details(soup: BeautifulSoup) -> Optional[Dict[str, str]]:
        """Extract detailed information from a single product page"""
        product_data = {
            "title": "",
            "description": "",
            "price": "",
            "images": [],
            "specifications": {},
            "reviews_summary": "",
            "availability": "",
            "sku": "",
            "brand": "",
            "category": "",
        }

        # Extract product title (usually h1)
        title_elem = soup.find("h1")
        if title_elem:
            product_data["title"] = title_elem.get_text(strip=True)

        # Extract price
        price_selectors = [
            {"class_": re.compile(r"product[-_]?price", re.I)},
            {"itemprop": "price"},
            {"data-price": True},
        ]
        for selector in price_selectors:
            price_elem = soup.find(["span", "div", "p"], **selector)
            if price_elem:
                product_data["price"] = price_elem.get_text(strip=True)
                break

        # Extract description
        desc_selectors = [
            {"class_": re.compile(r"product[-_]?description", re.I)},
            {"itemprop": "description"},
            {"id": re.compile(r"description", re.I)},
        ]
        for selector in desc_selectors:
            desc_elem = soup.find(["div", "p"], **selector)
            if desc_elem:
                product_data["description"] = desc_elem.get_text(strip=True)
                break

        # Extract images
        img_selectors = [
            {"class_": re.compile(r"product[-_]?image", re.I)},
            {"itemprop": "image"},
        ]
        for selector in img_selectors:
            for img in soup.find_all("img", **selector, src=True):
                product_data["images"].append(img.get("src", ""))

        # Extract specifications
        spec_elem = soup.find(
            ["div", "table"], class_=re.compile(r"spec|attribute|detail", re.I)
        )
        if spec_elem:
            specs = {}
            for row in spec_elem.find_all(["tr", "div"]):
                cells = row.find_all(["td", "span", "p"])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if key and value:
                        specs[key] = value
            product_data["specifications"] = specs

        # Extract brand
        brand_elem = soup.find(["span", "div", "a"], class_=re.compile(r"brand", re.I))
        if brand_elem:
            product_data["brand"] = brand_elem.get_text(strip=True)

        # Extract SKU
        sku_elem = soup.find(
            ["span", "div"], class_=re.compile(r"sku|product[-_]?code", re.I)
        )
        if sku_elem:
            product_data["sku"] = sku_elem.get_text(strip=True)

        # Extract availability
        avail_elem = soup.find(
            ["span", "div", "p"], class_=re.compile(r"stock|availability", re.I)
        )
        if avail_elem:
            product_data["availability"] = avail_elem.get_text(strip=True)

        return product_data if product_data["title"] else None

    @staticmethod
    def extract_main_content(soup: BeautifulSoup) -> str:
        """Extract main content area, avoiding navigation and UI elements"""

        # Priority order: try to find main content containers first
        main_content_selectors = [
            {"id": "main"},
            {"id": "content"},
            {"id": "main-content"},
            {"role": "main"},
            {"class_": re.compile(r"^main[-_]?content", re.I)},
            {"class_": re.compile(r"product[-_]?collection", re.I)},
            {"class_": re.compile(r"collection[-_]?grid", re.I)},
        ]

        for selector in main_content_selectors:
            main_elem = soup.find(["main", "div", "section"], **selector)
            if main_elem:
                return main_elem.get_text(separator=" ", strip=True)

        # Fallback: get body but remove known noise
        return soup.get_text(separator=" ", strip=True)

    @staticmethod
    def clean_product_text(text: str) -> str:
        """Clean extracted product text"""

        # Remove common e-commerce UI patterns
        noise_patterns = [
            # Navigation and UI
            r"(Sign In|Log in|Sign Up|Sign up|Create Account|My Account)\s*",
            r"(Add to cart|Add to bag|Add to wishlist|Quick view|Quick buy)\s*",
            r"(View Cart|Checkout|Continue shopping)\s*",
            r"(Sort by|Filter by|Showing \d+-\d+ of \d+)\s*",
            # Cookie and privacy
            r"(Cookie Policy|Privacy Policy|Terms of Service|Accept Cookies)\s*",
            # Newsletter/Marketing
            r"(Subscribe|Newsletter|Sign up for|Email signup)\s*",
            # Social media
            r"(Follow us|Share|Facebook|Twitter|Instagram|Pinterest)\s*",
            # Country/Region selectors (aggressive removal)
            r"(United States|United Kingdom|Australia|Canada|France|Germany|Italy|Spain|Japan|China|India|Brazil)\s+"
            * 3,
            r"(Shipping Country|Select Country|Choose Region)\s*",
            # Payment badges
            r"(Visa|Mastercard|PayPal|American Express|Discover)\s*",
            # Common footer text
            r"Copyright\s+©\s+\d{4}\s*",
            r"All rights reserved\s*",
        ]

        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove excessive country lists
        country_list = r"(?:Afghanistan|Albania|Algeria|Andorra|Angola|Argentina|Armenia|Australia|Austria|Azerbaijan|Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|Bosnia|Botswana|Brazil|Brunei|Bulgaria|Burkina|Burundi|Cambodia|Cameroon|Canada|Chad|Chile|China|Colombia|Congo|Costa Rica|Croatia|Cuba|Cyprus|Czechia|Denmark|Djibouti|Dominica|Ecuador|Egypt|El Salvador|Estonia|Ethiopia|Fiji|Finland|France|Gabon|Gambia|Georgia|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guyana|Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|Italy|Jamaica|Japan|Jordan|Kazakhstan|Kenya|Korea|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Lesotho|Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|Mauritania|Mauritius|Mexico|Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nepal|Netherlands|New Zealand|Nicaragua|Niger|Nigeria|Norway|Oman|Pakistan|Panama|Paraguay|Peru|Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Samoa|San Marino|Saudi Arabia|Senegal|Serbia|Seychelles|Singapore|Slovakia|Slovenia|Somalia|South Africa|Spain|Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|Togo|Trinidad|Tunisia|Turkey|Turkmenistan|Uganda|Ukraine|United Arab Emirates|United Kingdom|United States|Uruguay|Uzbekistan|Vanuatu|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe)"

        # Remove lines with 5+ countries
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            country_matches = re.findall(country_list, line, re.IGNORECASE)
            if len(country_matches) < 5:  # Keep line if less than 5 countries
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Clean up whitespace
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


class EnhancedEcommerceProductScraper:
    """Enhanced scraper optimized for e-commerce product pages"""

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        extract_product_links: bool = True,
    ):
        self.headless = headless
        self.timeout = timeout
        self.extract_product_links = extract_product_links
        self.browser: Optional[Browser] = None
        self.extractor = ProductContentExtractor()

    async def __aenter__(self):
        """Context manager entry"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    async def scrape_product_collection(self, url: str) -> Dict[str, any]:
        """
        Scrape product collection page with intelligent extraction

        Returns:
            Dict with:
                - 'text': Clean, structured text content
                - 'products': List of individual product data (if available)
                - 'product_urls': List of individual product URLs to scrape
                - 'error': Error message if any
        """
        if not self.browser:
            raise RuntimeError(
                "Browser not initialized. Use 'async with' context manager."
            )

        page: Optional[Page] = None
        try:
            logger.info(f"🛒 Scraping e-commerce page: {url}")

            # Create new page with realistic browser settings
            page = await self.browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            await page.set_viewport_size({"width": 1920, "height": 1080})

            # Navigate with fallback strategies
            html_content = await self._load_page_with_fallback(page, url)

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # STEP 1: Remove unwanted elements BEFORE extraction
            for element in soup(["script", "style", "iframe", "noscript"]):
                element.decompose()

            # Remove navigation, header, footer, aside
            for element in soup(["nav", "header", "footer", "aside"]):
                element.decompose()

            # Remove common UI clutter selectors
            clutter_selectors = [
                {"class_": re.compile(r"modal|popup|overlay|newsletter|cookie", re.I)},
                {"class_": re.compile(r"login|signup|account[-_]?menu", re.I)},
                {"class_": re.compile(r"cart|checkout|wishlist", re.I)},
                {"class_": re.compile(r"social[-_]?share|social[-_]?media", re.I)},
                {"class_": re.compile(r"breadcrumb|pagination", re.I)},
                {"class_": re.compile(r"filter|sort|sidebar", re.I)},
                {"id": re.compile(r"cart|checkout|wishlist", re.I)},
            ]

            for selector in clutter_selectors:
                for element in soup.find_all(**selector):
                    element.decompose()

            # STEP 2: Extract contact info
            contact_info = self._extract_contact_information(soup)

            # STEP 3: Detect if this is a single product page or collection page
            is_single_product = self._is_single_product_page(soup, url)

            structured_content = []
            products = []
            product_urls = []

            if is_single_product:
                # Extract single product details
                product = self.extractor.extract_single_product_details(soup)
                if product:
                    products.append(product)
                    structured_content.append(
                        self._format_single_product(product, contact_info)
                    )
            else:
                # Extract product collection
                products = self.extractor.extract_product_cards(soup)
                structured_content.append(
                    self._format_product_collection(soup, products, contact_info, url)
                )

                # Extract product URLs for individual scraping
                if self.extract_product_links:
                    for product in products:
                        if product["link"]:
                            absolute_url = urljoin(url, product["link"])
                            product_urls.append(absolute_url)

            final_text = "\n".join(structured_content)

            logger.info(
                f"✅ Successfully scraped e-commerce page: {len(final_text)} chars, "
                f"{len(products)} products found, {len(product_urls)} product URLs"
            )

            return {
                "text": final_text,
                "products": products,
                "product_urls": product_urls,
                "error": None,
            }

        except Exception as e:
            logger.error(f"❌ Failed to scrape {url}: {str(e)}")
            return {"text": "", "products": [], "product_urls": [], "error": str(e)}
        finally:
            if page:
                await page.close()

    def _is_single_product_page(self, soup: BeautifulSoup, url: str) -> bool:
        """Detect if this is a single product page or collection page"""
        # Check URL patterns
        single_product_patterns = [
            r"/product/",
            r"/item/",
            r"/p/",
            r"/products/[^/]+$",  # Single product at end
        ]

        for pattern in single_product_patterns:
            if re.search(pattern, url, re.I):
                return True

        # Check for schema.org Product markup
        if soup.find(attrs={"itemtype": re.compile(r"schema.org/Product$", re.I)}):
            return True

        # Check for single product indicators
        if soup.find("button", text=re.compile(r"add to cart", re.I)):
            return True

        return False

    def _format_single_product(self, product: Dict, contact_info: str) -> str:
        """Format single product data as structured text"""
        sections = []

        if contact_info:
            sections.append(f"CONTACT INFORMATION:\n{contact_info}\n")

        sections.append("PRODUCT DETAILS:")

        if product["title"]:
            sections.append(f"\nTitle: {product['title']}")

        if product["brand"]:
            sections.append(f"Brand: {product['brand']}")

        if product["sku"]:
            sections.append(f"SKU: {product['sku']}")

        if product["price"]:
            sections.append(f"Price: {product['price']}")

        if product["availability"]:
            sections.append(f"Availability: {product['availability']}")

        if product["description"]:
            sections.append(f"\nDescription:\n{product['description']}")

        if product["specifications"]:
            sections.append("\nSpecifications:")
            for key, value in product["specifications"].items():
                sections.append(f"  • {key}: {value}")

        if product["images"]:
            sections.append(f"\nImages: {len(product['images'])} available")

        return "\n".join(sections)

    def _format_product_collection(
        self, soup: BeautifulSoup, products: List[Dict], contact_info: str, url: str
    ) -> str:
        """Format product collection as structured text"""
        sections = []

        if contact_info:
            sections.append(f"CONTACT INFORMATION:\n{contact_info}\n")

        # Extract collection title
        title_elem = soup.find(["h1", "h2"])
        if title_elem:
            collection_title = title_elem.get_text(strip=True)
            sections.append(f"COLLECTION: {collection_title}\n")

        # Add collection description if available
        desc_elem = soup.find(
            ["div", "p"], class_=re.compile(r"collection[-_]?description", re.I)
        )
        if desc_elem:
            desc_text = desc_elem.get_text(strip=True)
            if desc_text and len(desc_text) > 20:
                sections.append(f"Description: {desc_text}\n")

        if products:
            sections.append(f"PRODUCTS ({len(products)} found):\n")
            for i, product in enumerate(products, 1):
                product_text = f"{i}. {product['title']}"

                if product["price"]:
                    product_text += f" - {product['price']}"

                if product["availability"]:
                    product_text += f" ({product['availability']})"

                sections.append(product_text)

                if product["description"]:
                    sections.append(f"   Description: {product['description']}")

                if product["sku"]:
                    sections.append(f"   SKU: {product['sku']}")

                if product["link"]:
                    absolute_url = urljoin(url, product["link"])
                    sections.append(f"   URL: {absolute_url}")

                sections.append("")  # Empty line between products

        return "\n".join(sections)

    async def _load_page_with_fallback(self, page: Page, url: str) -> str:
        """Load page with multiple fallback strategies"""
        strategies = [
            ("networkidle", self.timeout),
            ("domcontentloaded", self.timeout // 2),
            ("load", self.timeout // 3),
        ]

        last_error = None
        for wait_until, timeout in strategies:
            try:
                await page.goto(url, wait_until=wait_until, timeout=timeout)

                # Wait for product grid to load (common class names)
                try:
                    await page.wait_for_selector(
                        'div[class*="product"], div[class*="collection"], article[class*="product"]',
                        timeout=5000,
                    )
                except:
                    pass  # Not critical if selector not found

                # Additional wait for dynamic content
                await page.wait_for_timeout(2000)

                return await page.content()

            except PlaywrightTimeoutError as e:
                last_error = e
                logger.warning(f"Timeout with '{wait_until}', trying next strategy")
                continue

        raise last_error or Exception("Failed to load page with all strategies")

    def _extract_contact_information(self, soup: BeautifulSoup) -> str:
        """Extract contact information from page"""
        contact_parts = []
        all_text = soup.get_text()

        # Extract phone numbers
        phone_patterns = [
            r"\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        ]

        phone_numbers = set()
        for pattern in phone_patterns:
            matches = re.findall(pattern, all_text)
            for match in matches:
                cleaned = match.strip()
                digit_count = len(re.sub(r"\D", "", cleaned))
                if 10 <= digit_count <= 15:
                    digits_only = re.sub(r"\D", "", cleaned)
                    # Filter out dates and repeated digits
                    if len(set(digits_only)) > 1 and not re.search(
                        r"19\d{2}|20\d{2}", digits_only
                    ):
                        phone_numbers.add(cleaned)

        if phone_numbers:
            contact_parts.append(
                "📞 " + ", ".join(sorted(list(phone_numbers)[:3]))
            )  # Limit to 3

        # Extract emails
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = set(re.findall(email_pattern, all_text))

        # Filter noise emails
        emails = {e for e in emails if not re.search(r"example|test|noreply", e, re.I)}

        if emails:
            contact_parts.append(
                "📧 " + ", ".join(sorted(list(emails)[:3]))
            )  # Limit to 3

        return " | ".join(contact_parts) if contact_parts else ""


# Convenience function
async def scrape_ecommerce_url(url: str) -> Dict[str, any]:
    """Scrape e-commerce URL with enhanced extraction"""
    async with EnhancedEcommerceProductScraper() as scraper:
        return await scraper.scrape_product_collection(url)
