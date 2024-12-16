from argparse import ArgumentParser, Namespace

from src.data.data import *
from src.plots.centerpiece import create_interactive_event_map
from src.plots.maps import mention_bubbles_figure, migration_inflow_figure, migration_outflow_figure, refugee_outflow_figure
from src.analysis.timeseries import TSAnalyzer, precompute_country_embeddings, \
    plot_keyword_correlations_3d_with_dropdown

parser = ArgumentParser()

parser.add_argument('--migration_path', type=str, default="data/migration_bilateral.csv", help='Path to migration file. Absolute or relative path.')
parser.add_argument('--refugee_path', type=str, default="data/refugee_bilateral.csv", help='Path to refugee file. Absolute or relative path.')
parser.add_argument('--gdelt_path', default="data/gdelt.csv", type=str, help='Path to GDELT file. Absolute or relative path.')
parser.add_argument('--event_database_path', default="data/gdelt_event_database.csv", type=str, help='Path to GDELT event database file. Absolute or relative path.')
parser.add_argument('--print_debug', type=bool, default=False, help='Print debug information.')

def main(args: Namespace):
    gdelt = get_gdelt(args.gdelt_path)
    migration = get_migration(args.migration_path)
    refugee = get_refugee(args.refugee_path)
    event_database = GDELTEventDatabase(args.event_database_path, drop=False)
    centroids = load_country_centroids()

    test_plot_keyword_correlations(gdelt, migration, refugee, event_database)
    # test_analysis(gdelt, migration, refugee, event_database)
    # test_generate_data_and_plots(gdelt, migration, refugee, event_database, centroids)

def test_analysis(gdelt: GDELT, migration: Migration, refugee: Refugee, event_database: GDELTEventDatabase):
    ts_analyzer = TSAnalyzer(gdelt, migration, refugee, event_database)
    ts_analyzer.construct_keyword_series('USA')
    ts_analyzer.construct_in_migration_series('USA')
    ts_analyzer.contruct_out_migration_series('USA')
    refugee = ts_analyzer.construct_refugee_series('UKR', span=(2000, 2024))
    correlation_df = ts_analyzer.compute_spearman_correlations_for_keywords(
        country_iso='UKR',
        dataset=refugee,
        column='refugees',
        span=(2000, 2024)
    )
    fig = ts_analyzer.plot_keyword_correlations_3d(correlation_df)


def test_plot_keyword_correlations(gdelt: GDELT, migration: Migration, refugee: Refugee,
                                   event_database: GDELTEventDatabase):
    ts_analyzer = TSAnalyzer(gdelt, migration, refugee, event_database)

    # Define the list of countries you want to analyze
    # Use ISO3 country codes
    countries = ['USA', 'UKR']  # Add more as needed

    corr_data_by_country = {}

    for country_iso in countries:
        print(f"Processing country: {country_iso}")
        # Construct the refugee series (you can choose other datasets like migration)
        refugee_series = ts_analyzer.construct_refugee_series(country_iso, span=(2000, 2024))

        # Compute Spearman correlations between keywords and refugee counts
        corr_df = ts_analyzer.compute_spearman_correlations_for_keywords(
            country_iso=country_iso,
            dataset=refugee_series,
            column='refugees',
            span=(2000, 2024),
            num_keywords=100  # Adjust as needed
        )

        corr_data_by_country[country_iso] = corr_df

    # Precompute embeddings for all countries
    print("Precomputing embeddings for all countries...")
    country_data = precompute_country_embeddings(corr_data_by_country)

    if not country_data:
        print("No country data available for plotting.")
        return

    # Generate the interactive 3D plot with dropdown
    print("Generating interactive 3D plot with dropdown...")
    fig = plot_keyword_correlations_3d_with_dropdown(country_data)

    # Optionally, save the figure to an HTML file
    fig.write_html("keyword_correlations_3d_dropdown.html")
    print("3D plot saved to 'keyword_correlations_3d_dropdown.html'.")

    # Display the figure in a browser (optional)
    # Uncomment the following line if you want to automatically open the plot in your default web browser
    # pio.show(fig)


def test_generate_data_and_plots(gdelt: GDELT, migration: Migration, refugee: Refugee, event_database: GDELTEventDatabase, centroids):
    data = generate_plot_data(gdelt, migration, refugee, span=(2000, 2020), database=event_database)
    event_database.save_database()

    # Produce all the figures without displaying them for DEBUGGING
    # fig = create_interactive_event_map(data, centroids)
    # fig = mention_bubbles_figure(gdelt, event_database, span=(2000, 2020))
    # fig = migration_outflow_figure(migration)
    # fig = migration_inflow_figure(migration)
    # fig = refugee_outflow_figure(refugee)

if __name__ == '__main__':
    args: Namespace = parser.parse_args()
    main(args)