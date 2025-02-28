from pathlib import Path

import pandas as pd

mapping_files_numbers_to_names_shell = {  # Seems ok
    104: "tax-contribution-report-2023.pdf",
    # 105: "shell-sustainability-report-2023.pdf",
    108: "shell_annual_report_2018.pdf",
    109: "shell_annual_report_2019.pdf",
    112: "shell-annual-report-2022.pdf",
    113: "shell-annual-report-2023.pdf",
    114: "shell-climate-and-energy-transition-lobbying-report-2023.pdf",
}


mapping_files_numbers_to_names_thales = {  # to check
    208: "Thales - 2021 Full-Year results - slideshow.pdf",
    218: "Thales - 2021 Full-Year results - slideshow.pdf",
    209: "Thales - 2023 Full-Year results - slideshow_0.pdf",
    210: "Français - DEU 2021 020322_0.pdf",
    212: "Liste des participations hors de France 2023.12_FR.pdf",
    222: "Thales - Comptes consolidés au 31 décembre 2023.pdf",
    225: "Thales publie ses résultats annuels 2023 - Communiqué de presse - 5 mars 2024.pdf",
}


def get_shell_evaluation_df(csv_path: str | Path) -> pd.DataFrame:
    queries_responses = pd.read_csv(csv_path)
    for col in ["file_number", "page_number"]:
        queries_responses.loc[:, col] = queries_responses.loc[:, col].str.split(";")
    queries_responses = queries_responses.explode(["file_number", "page_number"])
    queries_responses_shell = (
        queries_responses.query("case == 'Shell' and file_number != '105'")
        .dropna(axis=1, how="all")
        .dropna(axis=0)
        .astype(
            {
                "file_number": int,
                "page_number": int,
            }
        )
    )
    queries_responses_shell.loc[:, "file"] = queries_responses_shell.file_number.map(
        mapping_files_numbers_to_names_shell
    )
    return queries_responses_shell
