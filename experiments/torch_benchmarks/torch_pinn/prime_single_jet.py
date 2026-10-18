"""STDE++ Section 6.2.4: genuine prime-based, single curved Taylor jet.

There is one *distinct prime per derivative occurrence*, including repeated
coordinate indices.  The curve uses ordinary power-series coefficients one:
    x(s) = x + sum_i e_{alpha[i]} s**p_i.
This is the paper's tangent construction rescaled to raw tangent
``x^(p_i) = p_i! e_{alpha[i]}``.  No floating-point factorial is formed.
Under the verified unique all-ones multiplicity condition, q[sum(p_i)] is
exactly D_alpha u; no subsequent huge-factorial division is needed.

This module prepares sparse inputs only.  Evaluation belongs to the same
Taylor backend used for the WDD comparison, with identical optimization rules.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from math import isqrt
import random
import time

import numpy as np
import torch

PAPER_URL = 'https://jmlr.org/papers/volume27/25-1474/25-1474.pdf'
VARIANT = 'STDE++ prime single curved jet (Section 6.2.4)'

# Table 1, printed p.14.  These are target contraction orders, not jet orders.
PAPER_PRIMES = {
    3: (5, 7, 11),
    4: (13, 17, 19, 31),
    5: (29, 31, 37, 41, 61),
    6: (71, 73, 79, 83, 103, 151),
    7: (127, 137, 157, 193, 197, 269, 277),
}

# Uniform seeded rejection in [m*k^2,2*m*k^2], for the predetermined sequence
# m=2,4,8,16,32,64, with at most 2000 draws per interval.  These are the first
# accepted draws of this search, NOT globally or interval-wise optimal primes.
EXTENDED_PRIMES = {
    8: (1093, 1291, 1321, 1367, 1451, 1693, 1973, 2039),
    9: (5581, 6217, 6599, 7577, 8147, 8287, 8419, 9539, 10103),
}
_RECORDED_SEARCH = {
    8: dict(seed=20261026, attempts=6015, precompute_wall_seconds=0.6407465001102537,
            search_intervals=[
                dict(lower=128, upper=256, attempts=2000, prime_count=23),
                dict(lower=256, upper=512, attempts=2000, prime_count=43),
                dict(lower=512, upper=1024, attempts=2000, prime_count=75),
                dict(lower=1024, upper=2048, attempts=15, prime_count=137)]),
    9: dict(seed=20261027, attempts=10082, precompute_wall_seconds=3.0055137081071734,
            search_intervals=[
                dict(lower=162, upper=324, attempts=2000, prime_count=29),
                dict(lower=324, upper=648, attempts=2000, prime_count=52),
                dict(lower=648, upper=1296, attempts=2000, prime_count=92),
                dict(lower=1296, upper=2592, attempts=2000, prime_count=167),
                dict(lower=2592, upper=5184, attempts=2000, prime_count=313),
                dict(lower=5184, upper=10368, attempts=82, prime_count=581)]),
}


def _is_prime(value):
    return value >= 2 and all(value % divisor for divisor in range(2, isqrt(value)+1))


def _counts_capped_two(orders):
    """Exact unbounded coin DP, using integer cumulative sums by residue.

    Each prime is processed as an unlimited coin type.  Capping a count at
    two preserves the sole distinction needed for uniqueness: zero, one, many.
    The padded columns are residue classes modulo the current prime, so a
    column cumulative sum is exactly the usual unbounded knapsack recurrence.
    """
    length = sum(orders)+1
    ways = np.zeros(length, dtype=np.int64)
    ways[0] = 1
    for prime in orders:
        rows = (length+prime-1)//prime
        padded = np.zeros(rows*prime, dtype=np.int64)
        padded[:length] = ways
        ways = np.minimum(np.cumsum(padded.reshape(rows, prime), axis=0,
                                    dtype=np.int64).reshape(-1)[:length], 2)
    return ways.astype(np.uint8)


@lru_cache(maxsize=None)
def _verified_certificate(orders):
    if not orders or tuple(sorted(orders)) != orders or len(set(orders)) != len(orders):
        raise ValueError('orders must be distinct increasing primes')
    if any(not isinstance(p, int) or isinstance(p, bool) or not _is_prime(p) for p in orders):
        raise ValueError('every order must be a prime integer')
    ways = _counts_capped_two(orders)
    count = int(ways[-1])
    # The all-ones vector is always a feasible solution to sum p_i*m_i=sum p_i.
    # Thus count==1 proves it is the sole nonnegative integer solution.
    subset_sums = {0}
    for prime in orders:
        subset_sums |= {degree+prime for degree in tuple(subset_sums)}
    certificate = dict(
        orders=list(orders), output_order=sum(orders), primes_distinct=True,
        solution_count=count, count_capped_at=2, unique=count == 1,
        uniqueness_status='unique_all_ones' if count == 1 else 'nonunique',
        unique_multiplicities=[1]*len(orders) if count == 1 else None,
        proof='Exact unbounded nonnegative-integer coin DP with counts capped at two',
        count_table_sha256=hashlib.sha256(ways.tobytes()).hexdigest(),
        subset_sum_count=len(subset_sums),
        target_relevant_orders_count=len(subset_sums) if count == 1 else None)
    if count == 1 and len(subset_sums) != 2**len(orders):
        raise AssertionError('unique all-ones solution must imply distinct subset sums')
    return certificate


def verify_unique_orders(orders):
    """JSON certificate for all nonnegative multiplicities, not just 0/1 subsets."""
    return deepcopy(_verified_certificate(tuple(orders)))


def search_prime_orders(k, *, seed=None, interval_multipliers=(2, 4, 8, 16, 32, 64),
                        attempts_per_interval=2000):
    """Reproduce the fixed uniform-prime rejection rule, with measured setup time.

    The returned wall time measures precomputation only; it is not an operator
    evaluation timing.  The accepted prime list is independently validated by
    ``verify_unique_orders`` before it is returned.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 2 or attempts_per_interval < 1:
        raise ValueError('k>=2 and a positive rejection limit are required')
    if seed is None:
        seed = 20261018+k
    rng = random.Random(seed)
    began = time.perf_counter()
    intervals = []
    attempts = 0
    for multiplier in interval_multipliers:
        if not isinstance(multiplier, int) or multiplier < 1:
            raise ValueError('positive integer interval multipliers are required')
        lower, upper = multiplier*k*k, 2*multiplier*k*k
        pool = [candidate for candidate in range(lower, upper+1) if _is_prime(candidate)]
        if len(pool) < k:
            intervals.append(dict(lower=lower, upper=upper, attempts=0, prime_count=len(pool)))
            continue
        for local_attempt in range(1, attempts_per_interval+1):
            orders = tuple(sorted(rng.sample(pool, k)))
            attempts += 1
            if int(_counts_capped_two(orders)[-1]) == 1:
                intervals.append(dict(lower=lower, upper=upper, attempts=local_attempt,
                                      prime_count=len(pool)))
                certificate = verify_unique_orders(orders)
                return dict(orders=list(orders), output_order=sum(orders), seed=seed,
                            attempts=attempts, search_intervals=intervals,
                            precompute_wall_seconds=time.perf_counter()-began,
                            certificate=certificate, selection='first accepted seeded draw',
                            interval_multipliers=list(interval_multipliers),
                            attempts_per_interval=attempts_per_interval,
                            globally_minimal=False)
        intervals.append(dict(lower=lower, upper=upper, attempts=attempts_per_interval,
                              prime_count=len(pool)))
    raise RuntimeError(f'no unique prime schedule found for k={k} after {attempts} draws')


@dataclass(frozen=True)
class PrimeSchedule:
    """One sparse curve; coefficients have shape [number of primes, dimension]."""
    orders: tuple[int, ...]
    coefficients: torch.Tensor
    output_order: int
    metadata: dict


def prime_schedule(alpha, dim, *, dtype=torch.float32, device='cpu'):
    """Prepare the faithful k-prime single-jet construction for D_alpha.

    ``alpha`` contains k repeated zero-based coordinate indices, NOT exponent
    counts.  Even a pure k-th derivative uses k distinct primes here.
    The common sparse backend accepts ``coefficients.unsqueeze(1)`` to add its
    singleton curve axis.  Its normalized output q_L is the requested partial.
    """
    alpha = tuple(alpha)
    if not alpha or not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError('a nonempty partial and positive input dimension are required')
    if any(not isinstance(axis, int) or isinstance(axis, bool) or axis < 0 or axis >= dim for axis in alpha):
        raise ValueError('partial coordinates must be nonnegative integers below dim')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('the prime single-jet inputs use real float32 or float64 coefficients')
    k = len(alpha)
    if k in PAPER_PRIMES:
        orders = PAPER_PRIMES[k]
        source = 'STDE++ Table 1'
        precomputation = dict(selection='published table', seed=None, attempts=None,
                              precompute_wall_seconds=None, globally_minimal=False)
    elif k in EXTENDED_PRIMES:
        orders = EXTENDED_PRIMES[k]
        source = 'seeded uniform prime rejection extending Section 6.2.4'
        precomputation = deepcopy(_RECORDED_SEARCH[k])
        precomputation.update(selection='first accepted seeded draw',
                              recorded_date='2026-10-18', globally_minimal=False,
                              interval_multipliers=[2, 4, 8, 16, 32, 64],
                              attempts_per_interval=2000,
                              wall_time_scope='recorded local prime-selection precomputation')
    else:
        raise ValueError('frozen validated prime schedules cover derivative orders 3 through 9')
    certificate = verify_unique_orders(orders)
    if not certificate['unique']:
        raise AssertionError('a frozen prime schedule failed exact uniqueness validation')
    coefficients = torch.zeros((k, dim), dtype=dtype, device=device)
    coefficients[torch.arange(k, device=device), torch.tensor(alpha, device=device)] = 1
    metadata = dict(variant=VARIANT, paper=PAPER_URL, paper_sections=['6.2.1', '6.2.4'],
                    derivative_order=k, alpha=list(alpha), prime_count=k, orders=list(orders),
                    output_order=sum(orders), curve_count=1, source=source,
                    repeated_coordinates_keep_distinct_occurrence_primes=True,
                    ordinary_input_coefficients='one-hot e_alpha[i] at degree p_i',
                    raw_input_tangents='p_i! e_alpha[i], represented implicitly without factorials',
                    output_extraction='normalized Taylor coefficient q_L equals D_alpha u',
                    floating_point_factorials_formed=False,
                    coordinate_or_jet_sampling_during_evaluation=False,
                    precomputation=precomputation, certificate=certificate)
    metadata['schedule_sha256'] = hashlib.sha256(json.dumps(
        dict(alpha=list(alpha), orders=list(orders), normalization='ordinary_coefficients_one'),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return PrimeSchedule(orders, coefficients, sum(orders), metadata)
