import data from "../data/dashboard_data.json";
import TeamGrid from "./components/TeamGrid";

// Calculate metrics on server
const getTeamMetrics = (teamName: string, teamData: any) => {
  const totalYards = teamData.team_stats.team_yards;
  const departingPlayers = teamData.players.filter((p: any) => p.status !== "Returning");
  const departingYards = departingPlayers.reduce((acc: number, p: any) => acc + p.stats.yards, 0);

  const lostProductionPct = totalYards > 0 ? (departingYards / totalYards) * 100 : 0;

  return {
    teamName,
    departingCount: departingPlayers.length,
    lostProductionPct,
    isCritical: lostProductionPct > 40
  };
};

export default function Home() {
  const teams = Object.keys(data).sort();
  const teamMetrics = teams.map(t => getTeamMetrics(t, (data as any)[t]));

  return (
    <div className="min-h-screen bg-background p-6 md:p-12 font-[family-name:var(--font-geist-sans)]">
      <header className="mb-10 max-w-7xl mx-auto border-b border-border pb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2">
          Roster Intelligence
        </h1>
        <p className="text-muted-foreground">
          Track transfer portal impact and system fit.
        </p>
      </header>

      <main className="max-w-7xl mx-auto">
        <TeamGrid teams={teamMetrics} />
      </main>
    </div>
  );
}
