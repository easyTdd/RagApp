from datetime import date
from typing import List, Optional

import requests
from bs4 import BeautifulSoup, Tag
from langchain_core.documents import Document
import re

def resolve_effective_dates(top_parts: List[Tag]) -> tuple[Optional[str], Optional[str]]:
    if len(top_parts) == 0:
        return (None, None)

    first_part = top_parts[0]

    # Find the first direct <p> tag whose text starts with "Suvestinė redakcija nuo"
    effective_period_text = None
    for child in first_part.children:
        if isinstance(child, Tag) and child.name == "p":
            text = child.get_text(strip=True)
            if text.startswith("Suvestinė redakcija nuo"):
                effective_period_text = text
                break

    if effective_period_text is None:
        return (None, None)
    
    # Extract dates using regex
    m = re.match(
        r"Suvestinė redakcija nuo (\d{4}-\d{2}-\d{2})(?: iki (\d{4}-\d{2}-\d{2}))?",
        effective_period_text
    )
    if m:
        effective_from = m.group(1)
        effective_to = m.group(2) if m.group(2) else None
        return (effective_from, effective_to)
    
    return (None, None)




class ESeimasHtmlLoader:
    """
    Iš e-seimas.lrs.lt HTML struktūros (div.WordSection1 > div#part...)
    padaro LangChain Document sąrašą.

    - Iki max_depth kuria atskirus dokumentus (vienas Document per div#part...).
    - Gildesnius nei max_depth part'us sukrauna į artimiausio viršaus dokumento tekstą.
    """

    def load(self, portal_url: str) -> List[Document]:
        """Atsisiunčia HTML iš URL ir grąžina dokumentų sąrašą."""
        try:
            document_url = portal_url.replace("portal/legalAct/lt/TAD", "rs/actualedition") + "/"

            html = self._download(document_url)
        except Exception as e:
            print(f"Failed to download HTML from {e}")
            raise

        soup = BeautifulSoup(html, "lxml")

        root = soup.find("div", class_="WordSection1")
        if root is None:
            raise ValueError("Neradau div.WordSection1 – HTML struktūra gal pasikeitė?")

        docs: List[Document] = []

        top_parts = root.find_all(self._is_part_div, recursive=False)

        (effective_from, effective_to) = resolve_effective_dates(top_parts)

        # top-level part'ai WordSection1 viduje
        for top_part in top_parts:
            self._walk_part(
                part_div=top_part,
                ancestors=[],
                parent_headings=[],
                docs=docs,
                url=portal_url,
                effective_from=effective_from,
                effective_to=effective_to
            )

        return docs

    # --- vidinės pagalbinės funkcijos ---

    def _download(self, url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://e-seimas.lrs.lt/",
            "Connection": "keep-alive",
        }
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text

    @staticmethod
    def _is_part_div(tag: Tag) -> bool:
        return (
            isinstance(tag, Tag)
            and tag.name == "div"
            and str(tag.get("id", "")).startswith("part")
        )

    def _walk_part(
        self,
        part_div: Tag,
        ancestors: List[Optional[str]],
        parent_headings: List[str],
        docs: List[Document],
        url: str,
        effective_from: Optional[str] = None,
        effective_to: Optional[str] = None
    ) -> None:
        """
        Apdoroja vieną div#part... mazgą.
        depth <= max_depth -> kuriamas atskiras Document.
        Vaikai:
          - jei jų depth <= max_depth -> jie bus atskiri Document (rekursyviai)
          - jei jų depth >  max_depth -> jų visas tekstas pridedamas prie šio Document.
        """
        # Get children parts
        children_parts: List[Tag] = [
            c for c in part_div.find_all(self._is_part_div, recursive=False)
        ] 

        # If not all children are separate documents, create a Document for this part
       
        text_chunks: List[str] = []
        text_chunks.extend(self._get_full_text(part_div))
        content = "\n".join(t for t in parent_headings + text_chunks if t).strip()

        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "url": url,
                        "reference": url + "#" + str(part_div.get("id")),
                        "heararchy": " > ".join(parent_headings),
                        "title": self._extract_part_div_title(part_div),
                        #"change_history": self._extract_change_history(part_div),
                        "effective_from": effective_from,
                        "effective_to": effective_to,
                    }
                )
            )

        # This is reached when all children are separate documents
        for child_part in children_parts:
            if self._is_part_an_artice_or_has_article(child_part):
                self._walk_part(
                    part_div=child_part,
                    ancestors=ancestors + [str(part_div.get("id"))],
                    parent_headings=parent_headings + [self._extract_part_div_title(part_div)],
                    docs=docs,
                    url=url,
                    effective_from=effective_from,
                    effective_to=effective_to
                )


    def _get_full_text(self, part_div: Tag) -> List[str]:
        """
        Paimti VISĄ tekstą iš šakos: visi p.MsoNormal žemiau šio div.
        Naudojama gilesniems nei max_depth part'ams sujungti į vieną gabalą.
        """

        result: List[str] = []

        for child in part_div.children:
            if not isinstance(child, Tag):
                continue
            if (
                child.name == "p"
                # and "MsoNormal" in (child.get("class") or [])
                # and child.find("i") is None # skip change history
            ):
                # Replace <sup> tags with ^{text} in the text
                # Replace <sup> tags with ^{text} in the text
                for sup in child.find_all("sup"):
                    sup.replace_with(f"^{sup.get_text(strip=True)}")
                # Replace <a> tags with text[href="url"]
                for a in child.find_all("a"):
                    link_text = a.get_text(strip=True)
                    href = a.get("href", "")
                    replacement = f'{link_text}[href="{href}"]' if href else link_text
                    a.replace_with(replacement)
                text = child.get_text("", strip=False)
                if text.strip():
                    result.append(text)
            elif child.name == "table" and "MsoNormalTable" in (child.get("class") or []):
                result.append(str(child))
            elif self._is_part_div(child) and not self._is_part_an_artice_or_has_article(child):
                result.extend(self._get_full_text(child))
            elif self._is_part_div(child):
                result.append(self._extract_part_div_title(child))
            else:
                result.append(child.get_text("", strip=False))

        return result
    
    def _extract_change_history(self, part_div: Tag) -> List[dict]:
        """
        Ištraukia pakeitimų istoriją iš part_div (p.MsoNormal su italic tekstu).
        """
        change_history_lines: List[dict] = []

        for child in part_div.children:
            if not isinstance(child, Tag):
                continue

            a_tag = child.find("a")

            if (
                child.name == "p"
                and "MsoNormal" in (child.get("class") or [])
                and child.find("i") is not None
                and a_tag is not None
                and a_tag.has_attr("href")
                and a_tag["href"]
            ):
                text = child.get_text("", strip=True)

                if not text:
                    continue

                # Extract dates from text
                acceptance_date = None
                announcement_date = None
                url = None
                law_no = a_tag.get_text(strip=True)

                # Pattern 1: TAR format
                m_tar = re.search(r"(\d{4}-\d{2}-\d{2}),\s*paskelbta TAR (\d{4}-\d{2}-\d{2})", text)
                if m_tar:
                    acceptance_date = m_tar.group(1)
                    announcement_date = m_tar.group(2)
                    url = a_tag["href"]
                else:
                    # Pattern 2: Žin. format
                    m_zin = re.search(r"(\d{4}-\d{2}-\d{2}),\s*Žin\.,.*\((\d{4}-\d{2}-\d{2})\)", text)
                    if m_zin:
                        acceptance_date = m_zin.group(1)
                        announcement_date = m_zin.group(2)

                change_history_lines.append({
                    "text": text,
                    "url": url,
                    "acceptance_date": acceptance_date,
                    "announcement_date": announcement_date,
                    "law_no": law_no
                })

            elif self._is_part_div(child) and not self._is_part_an_artice_or_has_article(child):
                nested_history = self._extract_change_history(child)
                if nested_history:
                    change_history_lines.extend(nested_history)

        # Return only distinct entries by law_no
        seen_law_nos = set()
        distinct_history = []
        for entry in change_history_lines:
            law_no = entry.get("law_no")
            if law_no and law_no not in seen_law_nos:
                seen_law_nos.add(law_no)
            distinct_history.append(entry)
        # Sort by announcement_date (None values last)
        distinct_history.sort(key=lambda x: (x.get("announcement_date") is None, x.get("announcement_date")))
        return distinct_history
    
    def _is_part_as_separate_document(self, part_div: Tag) -> bool:
        """
        Patikrinam ar straipsnis. Skaidom iki straipsnio lygio. 
        Rekursiškai tikrina ar šis yra straipsnis arba ar jo vaikai turi straipnsių.
        """
        all_p_tags = part_div.find_all("p", class_="MsoNormal")
        has_b_in_second_p = False
        if (
            len(all_p_tags) >= 2
            and all_p_tags[1].find("b") is not None
        ):
            has_b_in_second_p = True

        return has_b_in_second_p
    
    def _is_part_an_artice_or_has_article(self, part_div: Tag) -> bool:
        """
        Patikrinam ar straipsnis. Skaidom iki straipsnio lygio.
        """
        if re.match(r"^\d+(\.\d+|\(\d\)| \d)* straipsnis", self._extract_part_div_title(part_div).lower()):
            return True

        for child in part_div.children:
            if not isinstance(child, Tag):
                continue
            if self._is_part_div(child):
                if self._is_part_an_artice_or_has_article(child):
                    return True
                
        return False
    
    def _extract_part_div_title(self, part_div: Tag) -> str:
        """
        Ištraukia part_div antraštę (p.MsoNormal su bold tekstu).
        """
        def iter_direct_msonormal_p(div: Tag):
            for child in div.children:
                if not isinstance(child, Tag):
                    continue
                if child.name == "p":
                    classes = child.get("class") or []
                    if "MsoNormal" in classes:
                        yield child
                    else:
                        break  # Stop if p with other class
                else:
                    break  # Stop if any other element

        # header_p_tags = list(iter_direct_msonormal_p(part_div))

        # b_elements_texts = []
        # for p in header_p_tags:
        #     b_elements_texts.append(
        #         "".join(b.get_text(strip=False) for b in [bb for bb in p.find_all("b") if not bb.find_all("i")]).strip())
            
        # return " ".join(b for b in b_elements_texts if b).strip()
    
        header_p_tags = [p for p in iter_direct_msonormal_p(part_div) if p.find("b", recursive=True) and not p.find("i", recursive=True)]

        p_elements_texts = []
        for p in header_p_tags:
            p_elements_texts.append(
                p.get_text(" ", strip=True).strip()
            )
            
        return " ".join(b for b in p_elements_texts if b).strip()