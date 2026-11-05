"""File watcher for auto-triggering document processing."""
import time
import logging
from pathlib import Path
from typing import Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from config.config import DOCUMENTS_DIR

logger = logging.getLogger(__name__)


class DocumentHandler(FileSystemEventHandler):
    """Handler for document file system events."""
    
    def __init__(self, callback: Callable[[str], None], supported_extensions: set = {'.pdf', '.docx', '.doc'}):
        """
        Initialize the document handler.
        
        Args:
            callback: Function to call when a new document is detected
            supported_extensions: Set of supported file extensions
        """
        self.callback = callback
        self.supported_extensions = supported_extensions
        self.processed_files = set()
    
    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation event."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in self.supported_extensions:
            # Wait a bit for file to be fully written
            time.sleep(1)
            
            if file_path.exists() and str(file_path) not in self.processed_files:
                logger.info(f"New document detected: {file_path}")
                self.processed_files.add(str(file_path))
                try:
                    self.callback(str(file_path))
                except Exception as e:
                    logger.error(f"Error processing new document {file_path}: {e}")
                    self.processed_files.discard(str(file_path))
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification event."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix.lower() in self.supported_extensions:
            # Only process if not already processed
            if str(file_path) not in self.processed_files:
                logger.info(f"Modified document detected: {file_path}")
                self.processed_files.add(str(file_path))
                try:
                    self.callback(str(file_path))
                except Exception as e:
                    logger.error(f"Error processing modified document {file_path}: {e}")
                    self.processed_files.discard(str(file_path))


class DocumentWatcher:
    """Watch for new documents and trigger processing."""
    
    def __init__(self, documents_dir: str = str(DOCUMENTS_DIR), callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the document watcher.
        
        Args:
            documents_dir: Directory to watch for documents
            callback: Function to call when a new document is detected
        """
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.callback = callback
        self.observer = None
        self.handler = None
    
    def start(self, callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Start watching for new documents.
        
        Args:
            callback: Optional callback function (overrides the one set in __init__)
        """
        if callback:
            self.callback = callback
        
        if not self.callback:
            raise ValueError("No callback function provided")
        
        self.handler = DocumentHandler(self.callback)
        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.documents_dir), recursive=False)
        self.observer.start()
        logger.info(f"Started watching directory: {self.documents_dir}")
    
    def stop(self) -> None:
        """Stop watching for new documents."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching directory")
    
    def process_existing_files(self, callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Process all existing files in the directory.
        
        Args:
            callback: Optional callback function
        """
        if callback:
            self.callback = callback
        
        if not self.callback:
            raise ValueError("No callback function provided")
        
        logger.info(f"Processing existing files in {self.documents_dir}")
        for file_path in self.documents_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in {'.pdf', '.docx', '.doc'}:
                try:
                    self.callback(str(file_path))
                except Exception as e:
                    logger.error(f"Error processing existing file {file_path}: {e}")
