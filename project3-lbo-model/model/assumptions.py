"""
assumptions.py
--------------
Central dataclass for all LBO model assumptions.
Separates deal structure, operating projections, and exit assumptions
so scenarios can be constructed by swapping just the relevant fields.

All monetary values in $M. Rates as decimals (e.g., 0.08 = 8%).
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# SOFR assumption (current mid-curve forward, used for floating debt)
# ---------------------------------------------------------------------------
SOFR = 0.0530  # 5.30% — approximate 1M SOFR as of early 2026


@dataclass
class DebtTranche:
    """Represents one piece of the capital structure."""
    name: str
    principal: float          # $M drawn at close
    spread: float             # spread over SOFR (0.0 for fixed)
    fixed_rate: float         # fixed coupon (0.0 for floating)
    is_floating: bool         # True = SOFR + spread; False = fixed_rate
    term_years: int           # contractual maturity
    amort_pct: float          # required annual amortization as % of original principal
    is_revolver: bool = False  # True = revolving facility
    oid_pct: float = 0.0       # original issue discount (%)
    call_protection_yrs: int = 0  # years before callable at par

    @property
    def all_in_rate(self) -> float:
        return (SOFR + self.spread) if self.is_floating else self.fixed_rate

    @property
    def annual_amort(self) -> float:
        """Required cash amortization per year ($M)."""
        return self.principal * self.amort_pct

    @property
    def financing_fee(self) -> float:
        return self.principal * self.oid_pct


@dataclass
class DealAssumptions:
    """
    Master container for all LBO model inputs.

    Structured in three logical blocks:
      1. Entry / Transaction
      2. Operating Projections (per year)
      3. Exit
    """
    # -----------------------------------------------------------------------
    # 1. ENTRY / TRANSACTION ASSUMPTIONS
    # -----------------------------------------------------------------------
    entry_ebitda: float = 3_534.0        # LTM EBITDA used for entry pricing ($M)
    entry_ev_multiple: float = 10.5      # EV / LTM EBITDA entry multiple
    existing_net_debt: float = 6_512.0   # Target net debt being acquired ($M)
    existing_cash: float = 285.0         # Cash on target BS at close ($M)
    existing_gross_debt: float = 6_797.0 # Target gross debt ($M)
    advisory_fee_pct: float = 0.0055     # M&A advisory + legal as % of EV
    mgmt_fee_annual: float = 20.0        # Annual sponsor mgmt fee ($M/yr)
    hold_years: int = 5                  # Projected hold period

    # Opening BS (target company assets brought in, excl. goodwill & debt)
    opening_receivables: float = 224.0
    opening_inventory: float = 7_014.0
    opening_other_current: float = 500.0
    opening_ppe: float = 5_962.0
    opening_op_lease_rou: float = 8_089.0
    opening_other_lt_assets: float = 1_487.0
    opening_ap: float = 3_856.0
    opening_accrued: float = 784.0
    opening_current_lease: float = 1_002.0
    opening_other_current_liab: float = 389.0
    opening_lt_lease: float = 7_999.0
    opening_other_lt_liab: float = 1_724.0

    # Debt structure (five tranches)
    debt_tranches: List[DebtTranche] = field(default_factory=lambda: [
        DebtTranche(
            name="Revolver",
            principal=500.0,
            spread=0.0250, fixed_rate=0.0, is_floating=True,
            term_years=5, amort_pct=0.0, is_revolver=True, oid_pct=0.005,
        ),
        DebtTranche(
            name="Term Loan A",
            principal=4_000.0,
            spread=0.0275, fixed_rate=0.0, is_floating=True,
            term_years=5, amort_pct=0.05, is_revolver=False, oid_pct=0.01,
        ),
        DebtTranche(
            name="Term Loan B",
            principal=8_000.0,
            spread=0.0400, fixed_rate=0.0, is_floating=True,
            term_years=7, amort_pct=0.01, is_revolver=False, oid_pct=0.02,
        ),
        DebtTranche(
            name="Senior Secured Notes",
            principal=5_000.0,
            spread=0.0, fixed_rate=0.0725, is_floating=False,
            term_years=8, amort_pct=0.0, is_revolver=False, oid_pct=0.015,
            call_protection_yrs=3,
        ),
        DebtTranche(
            name="High Yield Notes",
            principal=3_500.0,
            spread=0.0, fixed_rate=0.0900, is_floating=False,
            term_years=10, amort_pct=0.0, is_revolver=False, oid_pct=0.02,
            call_protection_yrs=4,
        ),
    ])

    cash_sweep_pct: float = 1.00         # % of excess FCF swept to debt paydown
    min_cash_balance: float = 300.0      # Minimum cash to keep on BS ($M)

    # -----------------------------------------------------------------------
    # 2. OPERATING PROJECTIONS  (lists index 0 = Year 1, ..., n-1 = Year N)
    # -----------------------------------------------------------------------
    # Base revenue in LTM (actual); Year 1 revenue = LTM * (1 + growth[0])
    entry_revenue: float = 37_844.0

    revenue_growth: List[float] = field(default_factory=lambda:
        [0.040, 0.045, 0.050, 0.055, 0.055])

    ebitda_margin: List[float] = field(default_factory=lambda:
        [0.098, 0.102, 0.107, 0.112, 0.117])

    da_pct_revenue: List[float] = field(default_factory=lambda:
        [0.0355, 0.0345, 0.0335, 0.0330, 0.0325])

    capex_pct_revenue: List[float] = field(default_factory=lambda:
        [0.041, 0.039, 0.037, 0.035, 0.034])

    tax_rate: float = 0.228

    # Working capital drivers (days)
    dso: float = 2.2    # Days Sales Outstanding (receivables)
    dio: float = 96.1   # Days Inventory Outstanding (DG actual: 7014/26628*365)
    dpo: float = 52.8   # Days Payable Outstanding  (vs COGS)

    # -----------------------------------------------------------------------
    # 3. EXIT ASSUMPTIONS
    # -----------------------------------------------------------------------
    exit_ev_multiple: float = 10.5       # EV / LTM EBITDA exit multiple
    exit_year: int = 5                   # Year of exit (matches hold_years)

    # -----------------------------------------------------------------------
    # COMPUTED PROPERTIES
    # -----------------------------------------------------------------------
    @property
    def entry_ev(self) -> float:
        return self.entry_ebitda * self.entry_ev_multiple

    @property
    def entry_equity_value(self) -> float:
        return self.entry_ev - self.existing_net_debt

    @property
    def total_new_debt(self) -> float:
        return sum(t.principal for t in self.debt_tranches)

    @property
    def total_financing_fees(self) -> float:
        return sum(t.financing_fee for t in self.debt_tranches)

    @property
    def advisory_fees(self) -> float:
        return self.entry_ev * self.advisory_fee_pct

    @property
    def total_transaction_fees(self) -> float:
        return self.advisory_fees + self.total_financing_fees

    @property
    def sponsor_equity(self) -> float:
        """Equity check written by the sponsor at close ($M)."""
        uses = (self.entry_equity_value +
                self.existing_gross_debt +
                self.advisory_fees +
                self.total_financing_fees +
                self.min_cash_balance)
        return uses - self.total_new_debt

    @property
    def entry_leverage(self) -> float:
        """Entry Gross Debt / Entry EBITDA."""
        return self.total_new_debt / self.entry_ebitda

    @property
    def entry_net_leverage(self) -> float:
        """(Total Debt - Min Cash) / Entry EBITDA."""
        return (self.total_new_debt - self.min_cash_balance) / self.entry_ebitda


# ---------------------------------------------------------------------------
# Convenience: build scenario variants
# ---------------------------------------------------------------------------

def base_case() -> DealAssumptions:
    return DealAssumptions()


def bull_case() -> DealAssumptions:
    a = DealAssumptions()
    a.revenue_growth  = [r + 0.020 for r in a.revenue_growth]
    a.ebitda_margin   = [m + 0.010 for m in a.ebitda_margin]
    a.exit_ev_multiple = 12.0
    return a


def bear_case() -> DealAssumptions:
    a = DealAssumptions()
    a.revenue_growth  = [r - 0.025 for r in a.revenue_growth]
    a.ebitda_margin   = [m - 0.012 for m in a.ebitda_margin]
    a.exit_ev_multiple = 9.0
    return a
