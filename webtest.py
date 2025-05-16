from shiny import App, ui, render
import pandas as pd

df = pd.read_csv('/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/merged_summary_with_archetypes.csv')

# Removed static player list; we'll define it reactively
player_archetype = df['archetype_name'].unique().tolist()
app_ui = ui.page_fluid(
    ui.h2("CFB Quaterbacks stats"),
    ui.input_select("selected_archetype", "Select archetype",player_archetype),
    ui.output_table("player_table")
)

def server(input, output, session):
    @output
    @render.table
    def player_table():
        return (
            df[df["archetype_name"] == input.selected_archetype()]
            .sort_values(by="attempts", ascending=False)
        )

app = App(app_ui, server)
