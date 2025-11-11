"""
Enhanced Playwright scraper specifically optimized for e-commerce product pages.

This scraper intelligently extracts product information while filtering out UI noise.
"""

import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import Browser, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from ...utils.logging_config import get_logger
from .content_extractors import ContactExtractor
from .url_utils import log_domain_safely

logger = get_logger(__name__)


class ProductContentExtractor:
    """Intelligent product content extraction from e-commerce pages"""

    @staticmethod
    def extract_product_cards(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract individual product cards from listing pages"""
        products = []

        # Common product card selectors - expanded for better coverage
        # Shopify uses various structures, so we need comprehensive selectors
        product_selectors = [
            # Shopify-specific (comprehensive list)
            {"class_": re.compile(r"product[-_]?(card|item|tile|grid[-_]?item)", re.I)},
            {"class_": re.compile(r"collection[-_]?item", re.I)},
            {"class_": re.compile(r"grid-product", re.I)},
            {"class_": re.compile(r"product-card-wrapper", re.I)},
            {"class_": re.compile(r"product-item-wrapper", re.I)},
            {
                "class_": re.compile(r"grid__item", re.I)
            },  # Shopify grid items (very common)
            {"class_": re.compile(r"product-block", re.I)},
            {"class_": re.compile(r"card-wrapper", re.I)},  # Shopify Dawn theme
            {"class_": re.compile(r"product-card", re.I)},  # Generic product card
            {"class_": re.compile(r"product-item", re.I)},  # Generic product item
            # Data attributes (Shopify uses these extensively)
            {"data-product-id": True},
            {"data-product-handle": True},
            {"data-product-title": True},
            {"data-product-url": True},
            # Shopify-specific data attributes
            {"data-product": True},  # Some themes use just data-product
            # Schema.org markup
            {"itemtype": re.compile(r"schema.org/Product", re.I)},
            # WooCommerce
            {"class_": re.compile(r"product", re.I), "data-product-id": True},
            # Generic e-commerce patterns
            {"class_": re.compile(r"item|product-item", re.I)},
            {"class_": re.compile(r"product-tile|product-card", re.I)},
            # Additional patterns - look for elements containing product links
            # This is a fallback: if we find links to /products/, the parent might be a product card
        ]

        # First pass: try standard selectors
        found_elements = set()  # Track found elements to avoid duplicates
        for selector in product_selectors:
            product_elements = soup.find_all(["div", "article", "li", "a"], **selector)

            for element in product_elements:
                # Skip if we've already processed this element
                if id(element) in found_elements:
                    continue
                found_elements.add(id(element))
                product_data = {
                    "title": "",
                    "description": "",
                    "price": "",
                    "link": "",
                    "image": "",
                    "availability": "",
                    "sku": "",
                }

                # Extract product title - try multiple strategies
                title_elem = element.find(
                    ["h2", "h3", "h4", "a"],
                    class_=re.compile(r"title|name|product[-_]?name", re.I),
                )
                if title_elem:
                    product_data["title"] = title_elem.get_text(strip=True)

                # Fallback: check for data attributes or itemprop
                if not product_data["title"]:
                    # Check data-product-title
                    if element.get("data-product-title"):
                        product_data["title"] = element.get("data-product-title")
                    # Check itemprop="name"
                    elif element.find(attrs={"itemprop": "name"}):
                        product_data["title"] = element.find(
                            attrs={"itemprop": "name"}
                        ).get_text(strip=True)
                    # Check for any heading inside
                    elif element.find(["h1", "h2", "h3", "h4", "h5", "h6"]):
                        product_data["title"] = element.find(
                            ["h1", "h2", "h3", "h4", "h5", "h6"]
                        ).get_text(strip=True)

                # Extract product link - try multiple strategies
                link_elem = element.find("a", href=True)
                if link_elem:
                    href = link_elem.get("href", "")
                    # Make absolute URL if relative
                    if href and not href.startswith(("http://", "https://")):
                        # Will be made absolute later
                        product_data["link"] = href
                    else:
                        product_data["link"] = href

                # Also check for data attributes that might contain product URLs
                if not product_data["link"]:
                    for attr in ["data-product-url", "data-href", "data-link"]:
                        if element.get(attr):
                            product_data["link"] = element.get(attr)
                            break

                # Extract price - try multiple strategies
                price_elem = element.find(
                    ["span", "div", "p"], class_=re.compile(r"price", re.I)
                )
                if price_elem:
                    product_data["price"] = price_elem.get_text(strip=True)

                # Fallback: check for data attributes or itemprop
                if not product_data["price"]:
                    # Check data-price
                    if element.get("data-price"):
                        product_data["price"] = element.get("data-price")
                    # Check itemprop="price"
                    elif element.find(attrs={"itemprop": "price"}):
                        price_item = element.find(attrs={"itemprop": "price"})
                        product_data["price"] = price_item.get(
                            "content"
                        ) or price_item.get_text(strip=True)
                    # Check for money class (common in Shopify)
                    elif element.find(class_=re.compile(r"money|price-money", re.I)):
                        product_data["price"] = element.find(
                            class_=re.compile(r"money|price-money", re.I)
                        ).get_text(strip=True)

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

        # Second pass: If no products found, try finding product links and extracting from their parent containers
        # This handles cases where Shopify uses non-standard structures
        if not products:
            logger.debug(
                "No products found with standard selectors, trying link-based detection"
            )
            # Find all links that look like product URLs
            product_links = soup.find_all("a", href=re.compile(r"/product[s]?/", re.I))

            for link in product_links[
                :50
            ]:  # Limit to 50 to avoid too many false positives
                # Find the parent container that likely contains the product card
                parent = link.parent
                max_depth = 5  # Don't go too far up
                depth = 0

                while parent and depth < max_depth:
                    # Check if this parent looks like a product container
                    parent_class = parent.get("class", [])
                    parent_class_str = (
                        " ".join(parent_class).lower() if parent_class else ""
                    )

                    # Look for product-related classes
                    if any(
                        keyword in parent_class_str
                        for keyword in ["product", "item", "card", "grid"]
                    ):
                        # Skip if we've already processed this element
                        if id(parent) in found_elements:
                            break
                        found_elements.add(id(parent))

                        # Try to extract product data from this parent
                        product_data = {
                            "title": "",
                            "description": "",
                            "price": "",
                            "link": link.get("href", ""),
                            "image": "",
                            "availability": "",
                            "sku": "",
                        }

                        # Extract title from link text or nearby elements
                        link_text = link.get_text(strip=True)
                        if link_text and len(link_text) > 3:
                            product_data["title"] = link_text
                        else:
                            # Try to find title in parent
                            title_elem = parent.find(
                                ["h2", "h3", "h4", "span", "div"],
                                class_=re.compile(r"title|name|product", re.I),
                            )
                            if title_elem:
                                product_data["title"] = title_elem.get_text(strip=True)

                        # Extract price from parent
                        price_elem = parent.find(
                            ["span", "div", "p"],
                            class_=re.compile(r"price|money", re.I),
                        )
                        if price_elem:
                            product_data["price"] = price_elem.get_text(strip=True)

                        # Extract image
                        img_elem = parent.find("img", src=True)
                        if img_elem:
                            product_data["image"] = img_elem.get("src", "")

                        if product_data["title"]:
                            products.append(product_data)
                            break

                    parent = parent.parent
                    depth += 1

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
        """Clean extracted product text - AGGRESSIVE cleaning for UI noise"""

        # Remove common e-commerce UI patterns - MORE AGGRESSIVE
        noise_patterns = [
            # Navigation and UI
            r"(Sign In|Log in|Sign Up|Sign up|Create Account|My Account)",
            r"(Add to cart|Add to bag|Add to wishlist|Quick view|Quick buy)",
            r"(View Cart|Checkout|Continue shopping)",
            r"(Sort by|Filter by|Showing \d+-\d+ of \d+)",
            r"(Skip to content|Your cart is empty|Shopping cart Loading)",
            r"(Have an account|Log in to check out faster)",
            r"(Add note|Calculate shipping|Subtotal|Taxes and shipping calculated)",
            r"(Update|Check out|View Cart)",
            # Cookie and privacy
            r"(Cookie Policy|Privacy Policy|Terms of Service|Accept Cookies)",
            # Newsletter/Marketing
            r"(Subscribe|Newsletter|Sign up for|Email signup)",
            # Social media
            r"(Follow us|Share|Facebook|Twitter|Instagram|Pinterest)",
            # Country selectors
            r"(Shipping Country|Select Country|Choose Region|Province|Zip/Postal Code)",
            # Payment badges
            r"(Visa|Mastercard|PayPal|American Express|Discover)",
            # Common footer text
            r"Copyright\s+©\s+\d{4}",
            r"All rights reserved",
        ]

        for pattern in noise_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # Remove ALL country names (very aggressive)
        country_patterns = [
            r"Australia|Austria|Belgium|Canada|Czechia|Denmark|Finland|France|Germany|Hong Kong SAR|Ireland|Israel|Italy|Japan|Malaysia|Netherlands|New Zealand|Norway|Poland|Portugal|Singapore|South Korea|Spain|Sweden|Switzerland|United Arab Emirates|United Kingdom|United States",
            r"(Afghanistan|Albania|Algeria|Andorra|Angola|Argentina|Armenia|Australia|Austria|Azerbaijan|Bahamas|Bahrain|Bangladesh|Barbados|Belarus|Belgium|Belize|Benin|Bhutan|Bolivia|Bosnia|Botswana|Brazil|Brunei|Bulgaria|Burkina|Burundi|Cambodia|Cameroon|Canada|Chad|Chile|China|Colombia|Congo|Costa Rica|Croatia|Cuba|Cyprus|Czechia|Denmark|Djibouti|Dominica|Ecuador|Egypt|El Salvador|Estonia|Ethiopia|Fiji|Finland|France|Gabon|Gambia|Georgia|Germany|Ghana|Greece|Grenada|Guatemala|Guinea|Guyana|Haiti|Honduras|Hungary|Iceland|India|Indonesia|Iran|Iraq|Ireland|Israel|Italy|Jamaica|Japan|Jordan|Kazakhstan|Kenya|Korea|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Lesotho|Liberia|Libya|Liechtenstein|Lithuania|Luxembourg|Madagascar|Malawi|Malaysia|Maldives|Mali|Malta|Mauritania|Mauritius|Mexico|Moldova|Monaco|Mongolia|Montenegro|Morocco|Mozambique|Myanmar|Namibia|Nepal|Netherlands|New Zealand|Nicaragua|Niger|Nigeria|Norway|Oman|Pakistan|Panama|Paraguay|Peru|Philippines|Poland|Portugal|Qatar|Romania|Russia|Rwanda|Samoa|San Marino|Saudi Arabia|Senegal|Serbia|Seychelles|Singapore|Slovakia|Slovenia|Somalia|South Africa|Spain|Sri Lanka|Sudan|Suriname|Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|Togo|Trinidad|Tunisia|Turkey|Turkmenistan|Uganda|Ukraine|United Arab Emirates|United Kingdom|United States|Uruguay|Uzbekistan|Vanuatu|Venezuela|Vietnam|Yemen|Zambia|Zimbabwe)",
        ]

        for pattern in country_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

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

            # Capture contact information BEFORE mutating the DOM so we don't
            # accidentally delete phone/email blocks that live in headers/footers.
            contact_info = self._extract_contact_information(soup)

            # STEP 1: Remove unwanted elements BEFORE extraction
            for element in soup(["script", "style", "iframe", "noscript"]):
                element.decompose()

            # Remove navigation, header, footer, aside
            for element in soup(["nav", "header", "footer", "aside"]):
                element.decompose()

            # Remove common UI clutter selectors
            # BUT be careful not to remove product content areas
            clutter_selectors = [
                {"class_": re.compile(r"modal|popup|overlay|newsletter|cookie", re.I)},
                {"class_": re.compile(r"login|signup|account[-_]?menu", re.I)},
                {"class_": re.compile(r"cart|checkout|wishlist", re.I)},
                {"class_": re.compile(r"social[-_]?share|social[-_]?media", re.I)},
                {
                    "class_": re.compile(r"breadcrumb", re.I)
                },  # Keep pagination - might have product links
                # Don't remove filter/sort/sidebar - they might contain product info
                # Only remove if they're clearly UI-only (like filter buttons, not product containers)
                {"id": re.compile(r"cart|checkout|wishlist", re.I)},
            ]

            for selector in clutter_selectors:
                for element in soup.find_all(**selector):
                    # Don't remove if it contains product links or product data
                    has_product_link = element.find(
                        "a", href=re.compile(r"/product[s]?/", re.I)
                    )
                    has_product_data = element.find(
                        attrs={"data-product-id": True}
                    ) or element.find(attrs={"data-product-handle": True})
                    if not has_product_link and not has_product_data:
                        element.decompose()

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

                # Extract product URLs for individual scraping (do this BEFORE formatting)
                # This helps even if product cards weren't fully detected
                if self.extract_product_links:
                    # First, get links from detected products
                    for product in products:
                        if product["link"]:
                            # Make absolute URL
                            absolute_url = urljoin(url, product["link"])
                            # Clean URL (remove fragments, query params that aren't needed)
                            parsed = urlparse(absolute_url)
                            clean_url = (
                                f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                            )
                            if parsed.query and "variant" not in parsed.query.lower():
                                clean_url += f"?{parsed.query}"
                            product_urls.append(clean_url)

                    # Also extract product links from page (in case products list didn't capture them)
                    # Look for common product URL patterns in links
                    all_links = soup.find_all("a", href=True)
                    for link in all_links:
                        href = link.get("href", "")
                        if href:
                            # Check if it looks like a product URL
                            product_patterns = [
                                r"/product[s]?/",
                                r"/item[s]?/",
                                r"/p/",
                                r"/products/[^/]+$",
                            ]
                            for pattern in product_patterns:
                                if re.search(pattern, href, re.I):
                                    absolute_url = urljoin(url, href)
                                    if absolute_url not in product_urls:
                                        product_urls.append(absolute_url)
                                    break

                # Format the collection content
                formatted_text = self._format_product_collection(
                    soup, products, contact_info, url
                )

                # If no products were found but we have product URLs, add them to the text
                if not products and product_urls:
                    logger.info(
                        f"Found {len(product_urls)} product URLs but no product cards detected"
                    )
                    if formatted_text.strip():
                        formatted_text += (
                            f"\n\nPRODUCTS FOUND ({len(product_urls)} product URLs):\n"
                        )
                        for i, product_url in enumerate(
                            product_urls[:20], 1
                        ):  # Limit to 20 for display
                            formatted_text += f"{i}. {product_url}\n"
                    else:
                        # If formatted_text is empty, create basic structure
                        formatted_text = f"COLLECTION PAGE\n"
                        if contact_info:
                            formatted_text += (
                                f"CONTACT INFORMATION:\n{contact_info}\n\n"
                            )
                        formatted_text += (
                            f"PRODUCTS FOUND ({len(product_urls)} product URLs):\n"
                        )
                        for i, product_url in enumerate(product_urls[:20], 1):
                            formatted_text += f"{i}. {product_url}\n"

                structured_content.append(formatted_text)

            final_text = "\n".join(structured_content)

            # Log what we extracted for debugging
            logger.info(
                f"E-commerce extraction summary: products={len(products)}, "
                f"product_urls={len(product_urls)}, "
                f"structured_text_length={len(final_text)}, "
                f"is_single_product={is_single_product}"
            )

            # If no structured content was extracted, fall back to extracting main content
            if not final_text.strip():
                logger.warning(
                    f"No structured content extracted, falling back to main content extraction"
                )
                main_content = self.extractor.extract_main_content(soup)
                # Use less aggressive cleaning for fallback - we want to preserve content
                # Only do basic whitespace cleaning, not aggressive noise removal
                cleaned_content = main_content.strip()
                # Just normalize whitespace, don't remove content
                cleaned_content = re.sub(r"\s{3,}", " ", cleaned_content)
                cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)

                # Build simple structured output
                structured_content = []
                if contact_info:
                    structured_content.append(f"CONTACT INFORMATION:\n{contact_info}")

                # Add collection/page title
                title_elem = soup.find(["h1", "h2"])
                if title_elem:
                    structured_content.append(
                        f"TITLE: {title_elem.get_text(strip=True)}"
                    )

                # Add main content - be very lenient with length for collection pages
                # Collection pages might have minimal text but still be valid
                if cleaned_content and len(cleaned_content) > 10:  # Very low threshold
                    structured_content.append(f"\n{cleaned_content}")
                elif cleaned_content:
                    # Even if very short, include it - chunking will handle it
                    structured_content.append(f"\n{cleaned_content}")

                final_text = "\n".join(structured_content)

            # If still no content, get soup text as last resort
            if not final_text.strip():
                logger.warning(
                    "No content after all extractions, using soup text as fallback",
                    extra={
                        "url": log_domain_safely(url),
                        "products_found": len(products),
                        "product_urls_found": len(product_urls),
                    },
                )
                # Get ALL text from soup (we've already removed nav/header/footer)
                all_text = soup.get_text(separator=" ", strip=True)
                logger.info(f"Fallback soup text length: {len(all_text)} chars")

                # For last resort, use LESS aggressive cleaning - we need to preserve content
                # Only remove obvious UI noise, not everything
                # Remove only the most obvious noise patterns
                basic_noise = [
                    r"^\s*(Sign In|Sign Up|Log in|Create Account)\s*$",
                    r"^\s*(Add to cart|View Cart|Checkout)\s*$",
                ]
                for pattern in basic_noise:
                    all_text = re.sub(
                        pattern, "", all_text, flags=re.IGNORECASE | re.MULTILINE
                    )

                # Just normalize whitespace - don't remove content
                all_text = re.sub(r"\s{3,}", " ", all_text)
                all_text = re.sub(r"\n{3,}", "\n\n", all_text)
                all_text = all_text.strip()

                # Always use the text - even if short, chunking/filtering will handle it
                if len(all_text) > 0:
                    # Prepend contact info if we have it
                    if contact_info:
                        final_text = f"CONTACT INFORMATION:\n{contact_info}\n\nPAGE CONTENT:\n{all_text[:10000]}"
                    else:
                        final_text = f"PAGE CONTENT:\n{all_text[:10000]}"

                    logger.info(
                        f"Using fallback text: {len(final_text)} chars (original: {len(all_text)} chars)"
                    )
                else:
                    logger.error("Even fallback extraction returned empty text!")
                    final_text = ""

            # CRITICAL: Ensure we always return some text, even if minimal
            if not final_text or len(final_text.strip()) < 10:
                logger.error(
                    f"⚠️ E-commerce scraper returned minimal/empty text! "
                    f"Length: {len(final_text)}, Preview: {final_text[:100] if final_text else 'EMPTY'}",
                    extra={"url": log_domain_safely(url)},
                )
                # Create minimal text from URL and any found product URLs
                minimal_text = f"Collection page: {url}\n"
                if product_urls:
                    minimal_text += f"Found {len(product_urls)} product URLs:\n"
                    for i, purl in enumerate(product_urls[:10], 1):
                        minimal_text += f"{i}. {purl}\n"
                elif products:
                    minimal_text += f"Found {len(products)} products:\n"
                    for i, product in enumerate(products[:10], 1):
                        if product.get("title"):
                            minimal_text += f"{i}. {product['title']}\n"
                else:
                    minimal_text += "No products detected on this collection page."
                final_text = minimal_text
                logger.warning(
                    f"Created minimal fallback text: {len(final_text)} chars"
                )

            logger.info(
                f"✅ Successfully scraped e-commerce page: {len(final_text)} chars, "
                f"{len(products)} products found, {len(product_urls)} product URLs",
                extra={
                    "url": log_domain_safely(url),
                    "text_length": len(final_text),
                    "products_count": len(products),
                    "product_urls_count": len(product_urls),
                    "text_preview": final_text[:200] if final_text else "EMPTY",
                },
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
        else:
            # If no products found, try to extract some content from the page
            # Look for any text content in the main area
            main_content = soup.find(
                ["main", "div"], class_=re.compile(r"main|content|collection", re.I)
            )
            if main_content:
                # Get text but exclude navigation and UI elements
                text_content = main_content.get_text(separator=" ", strip=True)
                # Remove excessive whitespace
                text_content = re.sub(r"\s{3,}", " ", text_content)
                # Be very lenient - even short content is better than nothing
                if text_content and len(text_content) > 10:  # Very low threshold
                    sections.append(
                        f"PAGE CONTENT:\n{text_content[:2000]}\n"
                    )  # Limit to 2000 chars
            else:
                # Last resort: try to get any text from body
                body_text = soup.get_text(separator=" ", strip=True)
                body_text = re.sub(r"\s{3,}", " ", body_text)
                if body_text and len(body_text) > 10:
                    # Take first 2000 chars to avoid too much noise
                    sections.append(f"PAGE CONTENT:\n{body_text[:2000]}\n")

        return "\n".join(sections)

    async def _load_page_with_fallback(self, page: Page, url: str) -> str:
        """Load page with multiple fallback strategies"""
        strategies = [
            ("domcontentloaded", min(self.timeout, 20000)),  # Increased for Shopify
            ("load", min(self.timeout, 30000)),  # Increased for Shopify
            (
                "networkidle",
                min(self.timeout, 40000),
            ),  # Added networkidle for JS-heavy sites
        ]

        last_error = None
        for wait_until, timeout in strategies:
            try:
                await page.goto(url, wait_until=wait_until, timeout=timeout)

                # Wait longer for Shopify sites which often have lazy-loaded content
                is_shopify = await self._detect_shopify(page, page.url or url)
                wait_time = 5000 if is_shopify else 2000
                await page.wait_for_timeout(wait_time)

                # For Shopify, also wait for common content selectors
                if is_shopify:
                    try:
                        # Wait for common Shopify content selectors
                        # Also wait for product cards to appear (collection pages)
                        await page.wait_for_selector(
                            "main, [role='main'], .main-content, .product, .collection, "
                            ".grid__item, .product-card, .product-item, [data-product-id]",
                            timeout=10000,  # Increased timeout for collection pages
                        )
                        # Additional wait for lazy-loaded content
                        await page.wait_for_timeout(2000)
                    except Exception as e:
                        logger.debug(
                            f"Selector wait failed (non-critical): {e}",
                            extra={"url": log_domain_safely(url)},
                        )
                        # Still wait a bit for content to load
                        await page.wait_for_timeout(3000)
                        pass  # Continue even if selector doesn't appear

                return await page.content()

            except PlaywrightTimeoutError as e:
                last_error = e
                logger.warning(f"Timeout with '{wait_until}', trying next strategy")
                continue

        # If all strategies fail, return whatever content we can get
        logger.warning("All page load strategies failed, getting content anyway")
        try:
            return await page.content()
        except:
            raise last_error or Exception("Failed to load page")

    def _extract_contact_information(self, soup: BeautifulSoup) -> str:
        """
        Extract contact information from page.

        Uses the shared ContactExtractor utility to avoid code duplication.
        """
        return ContactExtractor.extract_from_soup(soup, include_emoji=True)

    def _guess_shopify_from_url(self, url: str) -> bool:
        if not url:
            return False
        url_lower = url.lower()
        shopify_keywords = [
            "shopify",
            ".myshopify.com",
            "/collections/",
            "/products/",
        ]
        return any(keyword in url_lower for keyword in shopify_keywords)

    async def _detect_shopify(self, page: Page, url: str) -> bool:
        """
        Detect Shopify storefronts even when the vanity domain does not
        include 'shopify'. Uses multiple heuristics so we can enable the
        longer waits/selectors that Shopify themes require.
        """
        if self._guess_shopify_from_url(url):
            return True

        detection_scripts = [
            "() => Boolean(window?.Shopify?.shop)",
            "() => Boolean(window?.Shopify?.routes)",
        ]

        for script in detection_scripts:
            try:
                if await page.evaluate(script):
                    return True
            except Exception:
                continue

        try:
            html = await page.content()
            shopify_indicators = [
                "cdn.shopify.com",
                "shopifycloud",
                "Shopify.shop",
            ]
            if any(indicator in html for indicator in shopify_indicators):
                return True
        except Exception:
            pass

        return False


# Convenience function
async def scrape_ecommerce_url(url: str) -> Dict[str, any]:
    """Scrape e-commerce URL with enhanced extraction"""
    async with EnhancedEcommerceProductScraper() as scraper:
        return await scraper.scrape_product_collection(url)
