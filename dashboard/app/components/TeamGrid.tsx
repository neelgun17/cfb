"use client";

import { useState } from "react";
import Link from "next/link";
import { Search, ArrowRight, Users, TrendingDown } from "lucide-react";
import { cn } from "@/lib/utils";

// Types
type TeamMetric = {
    teamName: string;
    departingCount: number;
    lostProductionPct: number;
    isCritical: boolean;
};

export default function TeamGrid({ teams }: { teams: TeamMetric[] }) {
    const [query, setQuery] = useState("");

    const filteredTeams = teams.filter((t) =>
        t.teamName.toLowerCase().includes(query.toLowerCase())
    );

    return (
        <div className="space-y-8">
            {/* Search Bar */}
            <div className="relative max-w-md mx-auto md:mx-0">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground w-4 h-4" />
                <input
                    type="text"
                    placeholder="Search teams..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="w-full bg-secondary/50 border border-border rounded-lg pl-10 pr-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/20 transition-all"
                />
            </div>

            {/* Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredTeams.map((team) => (
                    <Link
                        key={team.teamName}
                        href={`/team/${team.teamName}`}
                        className="group block bg-card border border-border rounded-lg p-5 hover:border-primary/30 transition-all"
                    >
                        <div className="flex justify-between items-start mb-4">
                            <h3 className="font-semibold text-lg">{team.teamName}</h3>
                            {team.isCritical && (
                                <div className="w-2 h-2 rounded-full bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.5)]" />
                            )}
                        </div>

                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between text-sm mb-1.5">
                                    <span className="text-muted-foreground">Lost Production</span>
                                    <span className={cn(
                                        "font-mono font-medium",
                                        team.lostProductionPct > 40 ? "text-orange-400" : "text-emerald-400"
                                    )}>
                                        {team.lostProductionPct.toFixed(1)}%
                                    </span>
                                </div>
                                {/* Minimal Progress Bar */}
                                <div className="h-1 w-full bg-secondary rounded-full overflow-hidden">
                                    <div
                                        className={cn(
                                            "h-full rounded-full",
                                            team.lostProductionPct > 40 ? "bg-orange-500" : "bg-emerald-500"
                                        )}
                                        style={{ width: `${team.lostProductionPct}%` }}
                                    />
                                </div>
                            </div>

                            <div className="flex items-center text-xs text-muted-foreground gap-2">
                                <Users className="w-3 h-3" />
                                {team.departingCount} Departures
                                <ArrowRight className="w-3 h-3 ml-auto opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-primary" />
                            </div>
                        </div>
                    </Link>
                ))}

                {filteredTeams.length === 0 && (
                    <div className="col-span-full text-center py-12 text-muted-foreground italic">
                        No teams found matching "{query}"
                    </div>
                )}
            </div>
        </div>
    );
}
