from pathlib import Path
from typing import List, Any, Dict
import csv
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
import psycopg2
from dotenv import load_dotenv

# Handle imports for both module and direct execution
try:
    from src.config import DatabaseConfig
    from src.logger import get_dataloader_logger
except ModuleNotFoundError:
    from config import DatabaseConfig
    from logger import get_dataloader_logger

load_dotenv()

# Initialize logger
logger = get_dataloader_logger()


# =====================================================================
# DATABASE HELPERS (uses centralized config)
# =====================================================================

def get_db_connection():
    """Get a PostgreSQL connection using centralized config"""
    return psycopg2.connect(**DatabaseConfig.get_connection_params())


def load_companies(data_dir: str) -> Dict[str, int]:
    """Load companies.csv and return mapping"""
    companies_file = Path(data_dir) / 'companies.csv'
    company_map = {}

    if not companies_file.exists():
        logger.warning(f"companies.csv not found at {companies_file}")
        return company_map

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        logger.info(f"Loading companies from {companies_file}")

        with open(companies_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                company_id = int(row['company_id'])
                company_name = row['company_name'].strip()
                domain = row.get('domain', '').strip()

                cur.execute(
                    """INSERT INTO companies (company_id, company_name, domain)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (company_name) DO NOTHING""",
                    (company_id, company_name, domain)
                )

                company_map[company_name.lower()] = company_id
                company_map[company_name.lower().replace(' ', '')] = company_id

                logger.info(f"  ✓ Company: {company_name} (ID: {company_id})")

        conn.commit()

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to load companies: {e}")
    finally:
        cur.close()
        conn.close()

    return company_map


def load_internal_employees_map(data_dir: str) -> Dict[int, int]:
    """Load internal_employees.csv and return employee_id -> company_id mapping"""
    emp_file = Path(data_dir) / 'internal_employees.csv'
    employee_company_map = {}

    if not emp_file.exists():
        logger.warning(f"internal_employees.csv not found at {emp_file}")
        return employee_company_map

    try:
        logger.info(f"Loading internal employees mapping from {emp_file}")

        with open(emp_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                emp_id = int(row['employee_id'])
                # Internal employees serve all companies, default to company 1
                employee_company_map[emp_id] = 1

        logger.info(f"Loaded {len(employee_company_map)} internal employee mappings")

    except Exception as e:
        logger.error(f"Failed to load internal employees: {e}")

    return employee_company_map


def extract_company_from_filename(filename: str, company_map: Dict[str, int]) -> int:
    """Extract company name from filename"""
    name_without_ext = Path(filename).stem
    name_lower = name_without_ext.lower()
    name_no_space_underscore = name_lower.replace(' ', '').replace('_', '')

    for company_name_key, company_id in company_map.items():
        company_lower = company_name_key.lower()
        company_no_space = company_lower.replace(' ', '').replace('_', '')

        if name_lower.startswith(company_lower):
            return company_id

        if company_no_space in name_no_space_underscore:
            return company_id

    return None


# =====================================================================
# GROUPED JSON LOADERS (group small docs by company)
# =====================================================================

def load_emails_grouped(data_dir: str, company_map: Dict[str, int]) -> List[Any]:
    """Load emails.json and GROUP by company_id"""
    email_file = Path(data_dir) / 'emails.json'
    documents = []

    if not email_file.exists():
        return documents

    try:
        logger.info(f"Loading emails from {email_file}")

        with open(email_file, 'r', encoding='utf-8') as f:
            emails = json.load(f)

        # Group emails by company_id
        emails_by_company = {}
        for email in emails:
            company_id = email.get('company_id')
            if company_id not in emails_by_company:
                emails_by_company[company_id] = []
            emails_by_company[company_id].append(email)

        # Create one document per company with all its emails
        for company_id, company_emails in emails_by_company.items():
            content = f"=== All Emails for Company ID {company_id} ===\n\n"

            for email in company_emails:
                content += f"""
--- Email ID: {email.get('email_id')} ---
From: {email.get('sender_name', 'Unknown')} <{email.get('sender_email', '')}>
To: {email.get('recipient_name', 'Unknown')} <{email.get('recipient_email', '')}>
Subject: {email.get('subject', '')}
Priority: {email.get('priority', '')}
Type: {email.get('type', '')}

Body:
{email.get('body', '')}

Response:
{email.get('answer', '')}

"""

            doc = Document(
                page_content=content,
                metadata={
                    'company_id': company_id,
                    'source': str(email_file),
                    'name': f'emails_company_{company_id}.json',
                    'type': 'json_emails',
                    'email_count': len(company_emails)
                }
            )
            documents.append(doc)

        logger.info(f"  ✓ Loaded and grouped {len(emails)} emails into {len(documents)} company documents")

    except Exception as e:
        logger.error(f"  ✗ Failed to load emails: {e}")

    return documents


def load_conversations_grouped(data_dir: str) -> List[Any]:
    """Load conversations.json and GROUP by company_id"""
    conv_file = Path(data_dir) / 'conversations.json'
    documents = []

    if not conv_file.exists():
        return documents

    try:
        logger.info(f"Loading conversations from {conv_file}")

        with open(conv_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        # Group conversations by company_id
        convs_by_company = {}
        for conv in conversations:
            company_id = conv.get('company_id')
            if company_id not in convs_by_company:
                convs_by_company[company_id] = []
            convs_by_company[company_id].append(conv)

        # Create one document per company with all its conversations
        for company_id, company_convs in convs_by_company.items():
            content = f"=== All Conversations for Company ID {company_id} ===\n\n"

            for conv in company_convs:
                content += f"--- Conversation ID: {conv.get('conversation_id')} ---\n"
                messages = conv.get('messages', [])
                for msg in messages:
                    content += f"{msg.get('sender_name', 'Unknown')}: {msg.get('content', '')}\n"
                content += "\n"

            doc = Document(
                page_content=content,
                metadata={
                    'company_id': company_id,
                    'source': str(conv_file),
                    'name': f'conversations_company_{company_id}.json',
                    'type': 'json_conversations',
                    'conversation_count': len(company_convs)
                }
            )
            documents.append(doc)

        logger.info(f"  ✓ Loaded and grouped {len(conversations)} conversations into {len(documents)} company documents")

    except Exception as e:
        logger.error(f"  ✗ Failed to load conversations: {e}")

    return documents


def load_notes_grouped(data_dir: str, employee_company_map: Dict[int, int]) -> List[Any]:
    """Load notes.json and GROUP by company_id"""
    notes_file = Path(data_dir) / 'notes.json'
    documents = []

    if not notes_file.exists():
        return documents

    try:
        logger.info(f"Loading notes from {notes_file}")

        with open(notes_file, 'r', encoding='utf-8') as f:
            notes = json.load(f)

        # Group notes by company_id
        notes_by_company = {}
        for note in notes:
            employee_id = note.get('employee_id')
            company_id = employee_company_map.get(employee_id, 1)

            if company_id not in notes_by_company:
                notes_by_company[company_id] = []
            notes_by_company[company_id].append(note)

        # Create one document per company with all its notes
        for company_id, company_notes in notes_by_company.items():
            content = f"=== All Notes for Company ID {company_id} ===\n\n"

            for note in company_notes:
                content += f"""--- Note ID: {note.get('note_id')} ---
Author: {note.get('employee_email', 'Unknown')}
Created: {note.get('created_at', '')}

{note.get('content', '')}

"""

            doc = Document(
                page_content=content,
                metadata={
                    'company_id': company_id,
                    'source': str(notes_file),
                    'name': f'notes_company_{company_id}.json',
                    'type': 'json_notes',
                    'note_count': len(company_notes)
                }
            )
            documents.append(doc)

        logger.info(f"  ✓ Loaded and grouped {len(notes)} notes into {len(documents)} company documents")

    except Exception as e:
        logger.error(f"  ✗ Failed to load notes: {e}")

    return documents


def load_internal_employees_into_db(data_dir: str) -> None:
    """Load internal_employees.csv into internal_employees table"""
    emp_file = Path(data_dir) / 'internal_employees.csv'

    if not emp_file.exists():
        logger.warning(f"internal_employees.csv not found at {emp_file}")
        return

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        logger.info(f"Loading internal employees into database from {emp_file}")

        with open(emp_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                emp_id = int(row['employee_id'])
                name = row['name'].strip()
                email = row['email'].strip()
                role = row['role'].strip()

                cur.execute(
                    """INSERT INTO internal_employees (emp_id, name, email, role)
                       VALUES (%s, %s, %s, %s)
                       ON CONFLICT (email) DO NOTHING""",
                    (emp_id, name, email, role)
                )
                count += 1

        conn.commit()
        logger.info(f"  ✓ Loaded {count} internal employees into database")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to load internal employees: {e}")
    finally:
        cur.close()
        conn.close()


def load_external_employees_into_db(data_dir: str) -> None:
    """Load external_employees.json into external_employees table"""
    emp_file = Path(data_dir) / 'external_employees.json'

    if not emp_file.exists():
        logger.warning(f"external_employees.json not found at {emp_file}")
        return

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        logger.info(f"Loading external employees into database from {emp_file}")

        with open(emp_file, 'r', encoding='utf-8') as f:
            employees_by_company = json.load(f)

        count = 0
        for company_id_str, employees in employees_by_company.items():
            company_id = int(company_id_str)

            for i, emp in enumerate(employees):
                ext_emp_id = emp.get('id', f"{company_id}_{i}")
                name = emp.get('name', '').strip()
                email = emp.get('email', '').strip()
                role = emp.get('role', 'External Employee').strip()

                if not email or not name:
                    continue

                cur.execute(
                    """INSERT INTO external_employees (ext_emp_id, company_id, name, email, role)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (email) DO NOTHING""",
                    (ext_emp_id, company_id, name, email, role)
                )
                count += 1

        conn.commit()
        logger.info(f"  ✓ Loaded {count} external employees into database")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to load external employees: {e}")
    finally:
        cur.close()
        conn.close()


def load_tasks_grouped(data_dir: str, employee_company_map: Dict[int, int]) -> List[Any]:
    """Load tasks.json and GROUP by company_id"""
    tasks_file = Path(data_dir) / 'tasks.json'
    documents = []

    if not tasks_file.exists():
        return documents

    try:
        logger.info(f"Loading tasks from {tasks_file}")

        with open(tasks_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

        # Group tasks by company_id
        tasks_by_company = {}
        for task in tasks:
            owner_id = task.get('owner_employee_id')
            company_id = employee_company_map.get(owner_id, 1)

            if company_id not in tasks_by_company:
                tasks_by_company[company_id] = []
            tasks_by_company[company_id].append(task)

        # Create one document per company with all its tasks
        for company_id, company_tasks in tasks_by_company.items():
            content = f"=== All Tasks for Company ID {company_id} ===\n\n"

            for task in company_tasks:
                content += f"""--- Task ID: {task.get('task_id')} ---
Title: {task.get('title', '')}
Owner: {task.get('owner_name', 'Unknown')}
Status: {task.get('status', '')}
Priority: {task.get('priority', '')}
Type: {task.get('type', '')}
Due: {task.get('due_date', '')}

Description:
{task.get('description', '')}

"""

            doc = Document(
                page_content=content,
                metadata={
                    'company_id': company_id,
                    'source': str(tasks_file),
                    'name': f'tasks_company_{company_id}.json',
                    'type': 'json_tasks',
                    'task_count': len(company_tasks)
                }
            )
            documents.append(doc)

        logger.info(f"  ✓ Loaded and grouped {len(tasks)} tasks into {len(documents)} company documents")

    except Exception as e:
        logger.error(f"  ✗ Failed to load tasks: {e}")

    return documents


# =====================================================================
# MAIN DOCUMENT LOADER
# =====================================================================

def load_all_documents(data_dir: str) -> List[Any]:
    """Load all documents: large ones individually, small ones grouped by company"""
    data_path = Path(data_dir).resolve()
    logger.info("=" * 80)
    logger.info(f"Loading documents from: {data_path}")
    logger.info("=" * 80)

    # Load companies and employee mappings into DB
    company_map = load_companies(data_dir)
    employee_company_map = load_internal_employees_map(data_dir)

    # Load employees into database tables
    load_internal_employees_into_db(data_dir)
    load_external_employees_into_db(data_dir)

    if not company_map:
        logger.error("No companies loaded. Cannot proceed.")
        return []

    documents = []

    # Load LARGE documents individually (PDFs, DOCX, TXT, XLSX, etc.)
    logger.info("Loading LARGE documents individually...")

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF files")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            full_text = "\n\n".join([page.page_content for page in pages])
            company_id = extract_company_from_filename(pdf_file.name, company_map)

            doc = Document(
                page_content=full_text,
                metadata={
                    'company_id': company_id,
                    'source': str(pdf_file),
                    'name': pdf_file.name,
                    'type': 'pdf',
                    'pages': len(pages)
                }
            )
            logger.info(f"  ✓ Loaded {pdf_file.name} (company_id: {company_id})")
            documents.append(doc)
        except Exception as e:
            logger.error(f"  ✗ Failed to load {pdf_file.name}: {e}")

    # Word/DOCX files
    docx_files = list(data_path.glob('**/*.docx'))
    logger.info(f"Found {len(docx_files)} Word files")
    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            for doc in loaded:
                company_id = extract_company_from_filename(docx_file.name, company_map)
                doc.metadata.update({
                    'company_id': company_id,
                    'source': str(docx_file),
                    'name': docx_file.name,
                    'type': 'docx'
                })
            logger.info(f"  ✓ Loaded {docx_file.name}")
            documents.extend(loaded)
        except Exception as e:
            logger.error(f"  ✗ Failed to load {docx_file.name}: {e}")

    # Text files
    txt_files = list(data_path.glob('**/*.txt'))
    logger.info(f"Found {len(txt_files)} TXT files")
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            for doc in loaded:
                company_id = extract_company_from_filename(txt_file.name, company_map)
                doc.metadata.update({
                    'company_id': company_id,
                    'source': str(txt_file),
                    'name': txt_file.name,
                    'type': 'txt'
                })
            logger.info(f"  ✓ Loaded {txt_file.name}")
            documents.extend(loaded)
        except Exception as e:
            logger.error(f"  ✗ Failed to load {txt_file.name}: {e}")

    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    logger.info(f"Found {len(xlsx_files)} Excel files")
    for xlsx_file in xlsx_files:
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            for doc in loaded:
                company_id = extract_company_from_filename(xlsx_file.name, company_map)
                doc.metadata.update({
                    'company_id': company_id,
                    'source': str(xlsx_file),
                    'name': xlsx_file.name,
                    'type': 'xlsx'
                })
            logger.info(f"  ✓ Loaded {xlsx_file.name}")
            documents.extend(loaded)
        except Exception as e:
            logger.error(f"  ✗ Failed to load {xlsx_file.name}: {e}")

    # CSV files (skip metadata CSVs)
    csv_files = [f for f in data_path.glob('**/*.csv')
                 if f.name not in ['companies.csv', 'internal_employees.csv']]
    logger.info(f"Found {len(csv_files)} CSV files")
    for csv_file in csv_files:
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            for doc in loaded:
                company_id = extract_company_from_filename(csv_file.name, company_map)
                doc.metadata.update({
                    'company_id': company_id,
                    'source': str(csv_file),
                    'name': csv_file.name,
                    'type': 'csv'
                })
            logger.info(f"  ✓ Loaded {csv_file.name}")
            documents.extend(loaded)
        except Exception as e:
            logger.error(f"  ✗ Failed to load {csv_file.name}: {e}")

    # Generic JSON files (skip data files like conversations, emails, notes, tasks)
    metadata_json_files = {'external_employees.json', 'conversations.json', 'emails.json', 'notes.json', 'tasks.json'}
    json_files = [f for f in data_path.glob('**/*.json') if f.name not in metadata_json_files]
    logger.info(f"Found {len(json_files)} JSON files")
    for json_file in json_files:
        try:
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema='.',
                text_content=False
            )
            loaded = loader.load()
            for doc in loaded:
                company_id = extract_company_from_filename(json_file.name, company_map)
                doc.metadata.update({
                    'company_id': company_id,
                    'source': str(json_file),
                    'name': json_file.name,
                    'type': 'json'
                })
            logger.info(f"  ✓ Loaded {json_file.name}")
            documents.extend(loaded)
        except Exception as e:
            logger.error(f"  ✗ Failed to load {json_file.name}: {e}")

    # Load employee data into database (not as documents)
    logger.info("Skipping employee documents - employees stored in tables instead")

    # Load SMALL documents GROUPED by company (emails, conversations, notes, tasks)
    logger.info("Loading SMALL documents GROUPED by company...")
    documents.extend(load_emails_grouped(data_dir, company_map))
    documents.extend(load_conversations_grouped(data_dir))
    documents.extend(load_notes_grouped(data_dir, employee_company_map))
    documents.extend(load_tasks_grouped(data_dir, employee_company_map))

    logger.info(f"Loaded {len(documents)} total documents (large docs individual + small docs grouped by company)")

    if documents:
        logger.info(f"Sample metadata from first document:")
        logger.info(f"  company_id: {documents[0].metadata.get('company_id')}")
        logger.info(f"  source: {documents[0].metadata.get('source')}")
        logger.info(f"  name: {documents[0].metadata.get('name')}")
        logger.info(f"  type: {documents[0].metadata.get('type')}")

    logger.info("=" * 80)
    return documents


if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"\nLoaded {len(docs)} documents.")
