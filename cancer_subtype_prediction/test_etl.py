import gzip
import os
import unittest
import tempfile
from etl import gunzip_file, download_file_from_url, extract_gene_names_from_gtf
# from roug_ml.utl.dowload_utils import download_file


class TestGunzipFile(unittest.TestCase):
    def test_gunzip_file_with_temp_directory(self):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Define paths
            source_path = os.path.join(temp_dir, "test.gz")
            dest_path = os.path.join(temp_dir, "test.txt")

            # Create a test gzip file
            content = b"Sample content for testing gzip decompression."
            with gzip.open(source_path, "wb") as f:
                f.write(content)

            # Decompress the file
            gunzip_file(source_path, dest_path)

            # Verify the content
            with open(dest_path, "rb") as f:
                content = f.read()
                self.assertEqual(
                    content, b"Sample content for testing gzip decompression."
                )


class TestDownloadFileFromURL(unittest.TestCase):
    def test_download_file_from_benchmark(self):
        # URL of a small, publicly available file
        test_url = (
            "https://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/destill.txt"
        )
        # Expected content of the benchmark file (optional)
        expected_content = b"top pressure"

        # Use a temporary directory to save the downloaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            destination_filename = "downloaded_file.txt"
            destination_path = os.path.join(temp_dir, destination_filename)

            # Download the file
            download_file_from_url(test_url, destination_path)
            # download_file(test_url, destination_path)

            # Check if the file exists
            self.assertTrue(os.path.exists(destination_path))

            # (Optional) Verify the content of the downloaded file
            with open(destination_path, "rb") as downloaded_file:
                content = downloaded_file.read()
                self.assertTrue(expected_content in content)


class TestExtractGeneNamesFromGTF(unittest.TestCase):
    def test_extract_gene_names_from_gtf(self):
        # Sample content for a mock GTF file
        mock_gtf_content = """
        # This is a comment line
        1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; gene_type "transcribed_unprocessed_pseudogene"; gene_name "DDX11L1"; transcript_id "ENST00000456328.2";
        1\tHAVANA\tgene\t29554\t31109\t.\t+\t.\tgene_id "ENSG00000243485.5"; gene_type "lncRNA"; gene_name "MIR1302-2HG"; transcript_id "ENST00000408384.1";
        1\tENSEMBL\tgene\t34554\t36081\t.\t-\t.\tgene_id "ENSG00000237613.2"; gene_type "protein_coding"; gene_name "FAM138A"; transcript_id "ENST00000417324.1";
        1\tENSEMBL\tgene\t52473\t54936\t.\t+\t.\tgene_id "ENSG00000268020.3"; gene_type "protein_coding"; gene_name "OR4G4P"; transcript_id "ENST00000606857.1";
        """.strip()

        # Path to the mock GTF file
        mock_gtf_path = "/tmp/mock.gtf"

        # Writing the mock GTF content to the file
        with open(mock_gtf_path, "w") as f:
            f.write(mock_gtf_content)

        # Call the function under test
        gene_names = extract_gene_names_from_gtf(mock_gtf_path)

        # Check that the list of gene names matches expected protein-coding genes
        expected_genes = ["FAM138A", "OR4G4P"]  # Corrected expected genes list
        self.assertEqual(
            set(gene_names),
            set(expected_genes),
            "Extracted gene names do not match expected protein-coding genes.",
        )


if __name__ == "__main__":
    unittest.main()
