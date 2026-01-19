
import Link from "next/link";
import { ArrowLeft, TrendingDown, TrendingUp, AlertCircle, Shield, CheckCircle2 } from "lucide-react";
import data from "../../../data/dashboard_data.json";
import { cn } from "@/lib/utils";

// Types
interface PlayerStats {
    yards: number;
    attempts: number;
    touchdowns: number;
}
interface PlayerImpact {
    yards_share: number;
    attempts_share: number;
}
interface Player {
    name: string;
    position: string;
    status: string;
    stats: PlayerStats;
    impact: PlayerImpact;
    archetype: string;
}
interface Recommendation {
    player: string;
    team_name: string;
    grades_pass?: number;
    formatted_grade?: number;
    archetype: string;
}
interface TeamData {
    team_stats: {
        team_yards: number;
        team_attempts: number;
        team_tds: number;
    };
    players: Player[];
    has_departures: boolean;
    team_archetype?: string;
    recommendations?: Recommendation[];
}

export async function generateStaticParams() {
    return Object.keys(data).map((team) => ({
        id: team,
    }));
}

export default async function TeamPage({ params }: { params: Promise<{ id: string }> }) {
    const resolvedParams = await params;
    const teamName = decodeURIComponent(resolvedParams.id);
    const teamData = (data as any)[teamName] as TeamData;

    if (!teamData) {
        return <div className="p-12 text-center text-muted-foreground">Team not found</div>;
    }

    const departingPlayers = teamData.players.filter((p) => p.status !== "Returning");
    const returningPlayers = teamData.players.filter((p) => p.status === "Returning");

    const totalYards = teamData.team_stats.team_yards;
    const departingYards = departingPlayers.reduce((acc, p) => acc + p.stats.yards, 0);
    const lostYardsPct = totalYards > 0 ? (departingYards / totalYards) * 100 : 0;

    return (
        <div className="min-h-screen bg-background p-6 md:p-12 font-[family-name:var(--font-geist-sans)]">
            <Link href="/" className="inline-flex items-center text-sm font-medium text-muted-foreground hover:text-foreground mb-8 transition-colors">
                <ArrowLeft className="w-4 h-4 mr-2" /> Back to Dashboard
            </Link>

            <div className="max-w-6xl mx-auto space-y-12">
                {/* Header Section */}
                <header className="border-b border-border pb-8">
                    <h1 className="text-4xl font-bold tracking-tight mb-2">{teamName}</h1>
                    <div className="flex items-center gap-4 text-muted-foreground text-sm">
                        <span className="flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-secondary border border-border">
                            <Shield className="w-3 h-3" />
                            System: {teamData.team_archetype || "Unknown"}
                        </span>
                        <span>•</span>
                        <span>{teamData.team_stats.team_yards.toLocaleString()} Total Passing Yards</span>
                    </div>
                </header>

                {/* Key Metric: Impact */}
                <section className="bg-secondary/30 rounded-xl p-8 border border-border flex flex-col md:flex-row items-center justify-between gap-6">
                    <div>
                        <h2 className="text-lg font-semibold mb-1">Pass Production Lost</h2>
                        <p className="text-muted-foreground text-sm max-w-sm">
                            Percentage of total passing yards from departing players (Draft or Portal).
                        </p>
                    </div>
                    <div className="flex items-end gap-3">
                        <span className={cn(
                            "text-5xl font-mono font-bold tracking-tighter",
                            lostYardsPct > 40 ? "text-orange-500" : "text-emerald-500"
                        )}>
                            {lostYardsPct.toFixed(1)}%
                        </span>
                    </div>
                </section>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
                    {/* Left Col: Roster (7 cols) */}
                    <div className="lg:col-span-7 space-y-10">
                        {/* Departures */}
                        <section>
                            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4 flex items-center gap-2">
                                <AlertCircle className="w-4 h-4" /> Departures
                            </h3>
                            {departingPlayers.length === 0 ? (
                                <div className="text-sm text-muted-foreground italic border-l-2 border-border pl-4">No major QB departures.</div>
                            ) : (
                                <div className="space-y-4">
                                    {departingPlayers.map((p) => (
                                        <div key={p.name} className="flex justify-between items-start border-b border-border pb-4 last:border-0">
                                            <div>
                                                <div className="font-semibold text-lg">{p.name}</div>
                                                <div className="text-xs text-muted-foreground flex gap-2 mt-1">
                                                    <span className="px-1.5 py-0.5 bg-secondary rounded">{p.status}</span>
                                                    <span>{p.archetype}</span>
                                                </div>
                                            </div>
                                            <div className="text-right">
                                                <div className="font-mono font-medium">{p.stats.yards.toLocaleString()}</div>
                                                <div className="text-xs text-muted-foreground">Yards</div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </section>

                        {/* Returning */}
                        <section>
                            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">Returning Roster</h3>
                            {returningPlayers.length === 0 ? (
                                <div className="text-sm text-muted-foreground italic border-l-2 border-border pl-4">No returning QBs with significant stats.</div>
                            ) : (
                                <ul className="space-y-2">
                                    {returningPlayers.slice(0, 5).map((p) => (
                                        <li key={p.name} className="flex justify-between items-center text-sm py-2 px-3 rounded-md hover:bg-secondary/40 transition-colors">
                                            <span>{p.name}</span>
                                            <span className="font-mono text-muted-foreground">{p.stats.yards} yds</span>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </section>
                    </div>

                    {/* Right Col: Solutions / Recommendations (5 cols) */}
                    <div className="lg:col-span-5">
                        <div className="sticky top-8 space-y-6">
                            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4 flex items-center gap-2">
                                <CheckCircle2 className="w-4 h-4 text-emerald-500" /> System Matches
                            </h3>

                            {(teamData.recommendations && teamData.recommendations.length > 0) ? (
                                <div className="space-y-3">
                                    {teamData.recommendations.map((rec, i) => (
                                        <div key={rec.player} className="bg-card border border-border rounded-lg p-4 hover:border-emerald-500/30 hover:shadow-sm transition-all group">
                                            <div className="flex items-center gap-3 mb-2">
                                                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-secondary text-xs font-mono font-medium text-muted-foreground group-hover:bg-emerald-500/10 group-hover:text-emerald-500 transition-colors">
                                                    {i + 1}
                                                </span>
                                                <span className="font-semibold">{rec.player}</span>
                                            </div>
                                            <div className="text-sm text-muted-foreground flex justify-between">
                                                <span>{rec.team_name}</span>
                                                <span className="text-xs opacity-75">{rec.archetype}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="p-6 border border-dashed border-border rounded-lg text-center">
                                    <p className="text-muted-foreground text-sm">No specific system matches found.</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
