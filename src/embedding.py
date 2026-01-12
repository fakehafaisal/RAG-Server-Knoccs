import re
from typing import List, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-mpnet-base-v2", 
                 chunk_size: int = 1024, chunk_overlap: int = 256):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
    
    def clean_pdf_text(self, text: str) -> str:
        """Clean common PDF extraction artifacts"""
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)  # remove page numbers
        text = re.sub(r'\n{3,}', '\n\n', text)  # remove excessive whitespace
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # remove hyphenation at line breaks
        
        # remove lines that are mostly symbols and no words
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            alphanum = sum(c.isalnum() for c in line)
            total = len(line.strip())
            if total > 0 and alphanum / total > 0.3:  # At least 30% alphanumeric
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def is_section_header(self, line: str, next_line: str = "") -> bool:
        """Detect if a line is likely a section header"""
        line = line.strip()
        
        if not line or len(line) > 150:
            return False

        # Numbered sections: "1 Introduction", "2.1 Methods", etc.
        if re.match(r'^\d+\.?\d*\s+[A-Z]', line):
            return True
        
        # Short ALL CAPS lines
        if line.isupper() and 5 < len(line) < 50:
            return True
        
        # Lines ending with colon
        if line.endswith(':') and len(line) < 100:
            return True
        
        # Title Case headers
        words = line.split()
        if len(words) <= 10:
            cap_words = sum(1 for w in words if w and w[0].isupper())
            if cap_words / len(words) > 0.6:
                return True
        
        # Check if next line is paragraph text
        if next_line:
            next_clean = next_line.strip()
            if next_clean and (next_clean[0].islower() or len(next_clean.split()) > 5):
                if len(line.split()) <= 8:
                    return True
        
        return False
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Chunk documents with improved header detection and cleaning.
        Preserves company_id and other critical metadata through chunking.
        """
        all_chunks = []
        
        # Use semantic splitter with multiple separators
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",  # Multiple newlines
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                ""
            ],
            length_function=len,
        )
        
        for doc_idx, doc in enumerate(documents):
            # Extract critical metadata BEFORE chunking
            company_id = doc.metadata.get('company_id')
            source = doc.metadata.get('source', f'doc_{doc_idx}')
            doc_name = doc.metadata.get('name', 'unknown')
            doc_type = doc.metadata.get('type', 'unknown')
            
            if company_id is None:
                print(f"[WARNING] Document '{doc_name}' has no company_id assigned!")
                continue
            
            # Clean the PDF text first
            cleaned_text = self.clean_pdf_text(doc.page_content)
            
            # Split into initial chunks
            # Pass metadata to preserve company_id through LangChain
            initial_chunks = text_splitter.create_documents(
                [cleaned_text],
                metadatas=[{
                    'company_id': company_id,
                    'source': source,
                    'name': doc_name,
                    'type': doc_type
                }]
            )
            
            # Post-process chunks: add section context
            lines = cleaned_text.split('\n')
            current_section = "Document Start"
            
            # Detect sections
            for i, line in enumerate(lines):
                next_line = lines[i+1] if i+1 < len(lines) else ""
                if self.is_section_header(line, next_line):
                    current_section = line.strip()
            
            # Add chunks with enriched metadata
            for chunk_idx, chunk in enumerate(initial_chunks):
                text = chunk.page_content.strip()
                
                # Skip if chunk is too short or mostly symbols
                if len(text) < 50:
                    continue
                
                alphanum_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
                if alphanum_ratio < 0.6:  # Less than 60% alphanumeric + spaces
                    continue
                
                # Update metadata with all required fields
                chunk.metadata.update({
                    'company_id': company_id,  # Ensure company_id is preserved
                    'source': source,
                    'name': doc_name,
                    'type': doc_type,
                    'chunk_id': f"{doc_idx}_{chunk_idx}",
                    'document': source,  # Used for mapping to document in vectorstore
                    'section': current_section
                })
                
                all_chunks.append(chunk)
        
        print(f"[INFO] Created {len(all_chunks)} cleaned chunks from {len(documents)} documents")
        
        # Validate that all chunks have company_id
        chunks_without_company = [c for c in all_chunks if c.metadata.get('company_id') is None]
        if chunks_without_company:
            print(f"[WARNING] {len(chunks_without_company)} chunks missing company_id!")
        
        return all_chunks
    
    def embed_chunks(self, chunks: List[Any]):
        """Embed chunks using sentence transformers"""
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    