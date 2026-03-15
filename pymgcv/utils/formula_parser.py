"""Formula parser for GAM specifications.

Parses formulas like 'y ~ s(x1) + s(x2, k=10) + te(x3, x4) + x5'
into structured smooth specifications and parametric terms.

Module exports:
    - Formula: Main formula parser class
    - SmoothSpec: Specification for a single smooth term
    - ParametricSpec: Parametric (non-smooth) term specification
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SmoothSpec:
    """Specification for a single smooth term in a GAM.

    Attributes:
        term_type: Basis type ('s', 'te', 'ti', 're', 'tp', 'cc', 't2', etc.)
        variables: List of variable names involved in the smooth.
        basis: Basis system ('tp', 'cs', 'cc', 're', 'bs', etc.). Default 'tp'.
        k: Basis dimension (number of knots). None means auto-select.
        m: Smoothness order for splines (e.g., m=(2,1) for cubic).
        by_variable: Name of by-variable for varying-coefficient models.
        fx: If True, basis is fixed (unpenalized).
        bs_args: Additional basis-specific arguments as dict.
        label: Optional descriptive label.

    Example:
        >>> spec = SmoothSpec(term_type='s', variables=['age'], basis='tp', k=10)
        >>> spec.label = 's(age, k=10)'
    """

    term_type: str  # 's', 'te', 'ti', 're', 'tp', 'cc', 't2'
    variables: list[str]
    basis: str = 'tp'  # thin plate regression spline
    k: int | None = None
    m: tuple[int, int] | None = None
    by_variable: str | None = None  # for s(x, by=group)
    fx: bool = False  # fixed (unpenalized) basis
    bs_args: dict[str, Any] = field(default_factory=dict)
    label: str = ''

    def __post_init__(self) -> None:
        """Validate and construct label if not provided."""
        if not self.label:
            var_str = ', '.join(self.variables)
            k_str = f', k={self.k}' if self.k is not None else ''
            by_str = f', by={self.by_variable}' if self.by_variable else ''
            self.label = f'{self.term_type}({var_str}, basis={self.basis}{k_str}{by_str})'


@dataclass
class ParametricSpec:
    """Specification for a parametric (linear) term in a GAM.

    Attributes:
        variables: List of variable names (typically single variable).
        interaction: If True, term is expressed with interactions.
        label: Descriptive label.
    """

    variables: list[str]
    interaction: bool = False
    label: str = ''

    def __post_init__(self) -> None:
        """Construct label if not provided."""
        if not self.label:
            self.label = ' * '.join(self.variables) if self.interaction else self.variables[0]


class FormulaParser:
    """Parser for GAM formulas in extended Wilkinson notation.

    Parses formulas of the form:
        y ~ s(x1) + s(x2, k=10) + te(x3, x4) + x5 + factor(region)

    Supports:
        - Smooth terms: s(x), te(x1,x2), ti(x1,x2), re(group), tp(x)
        - Parametric terms: x, interactions x1*x2
        - Functions: log(x), factor(x), offset(x)
        - Basis specifications: s(x, basis='cs', k=15)

    Attributes:
        formula: Original formula string.
        response: Response variable name(s).
        smooth_terms: List of SmoothSpec objects.
        parametric_terms: List of ParametricSpec objects.
        parametric_names: List of parametric variable names.
        offset_term: Offset variable name if present.
        function_terms: Dict mapping variable names to function calls.
    """

    # Regex patterns — match s, te, ti, t2, re terms
    SMOOTH_PATTERN = re.compile(
        r'(t2|te|ti|re|s)\s*\(\s*([^)]+)\s*\)',
        re.IGNORECASE
    )
    FUNCTION_PATTERN = re.compile(
        r'(\w+)\s*\(\s*([^)]+)\s*\)'
    )
    VARIABLE_PATTERN = re.compile(r'[a-zA-Z_]\w*')

    def __init__(self, formula: str) -> None:
        """Initialize parser with formula string.

        Args:
            formula: Formula string, e.g., 'y ~ s(x1) + s(x2) + x3'

        Raises:
            ValueError: If formula is malformed or missing response/predictor.
        """
        self.formula = formula.strip()
        self.response: str = ''
        self.smooth_terms: list[SmoothSpec] = []
        self.parametric_terms: list[ParametricSpec] = []
        self.parametric_names: list[str] = []
        self.offset_term: str | None = None
        self.function_terms: dict[str, str] = {}

        self._parse()

    def _parse(self) -> None:
        """Parse formula into components."""
        # Split response ~ predictors
        if '~' not in self.formula:
            raise ValueError(f'Invalid formula: {self.formula}. Must contain "~".')

        parts = self.formula.split('~')
        if len(parts) != 2:
            raise ValueError(f'Invalid formula: {self.formula}. Exactly one "~" required.')

        self.response = parts[0].strip()
        rhs = parts[1].strip()

        if not self.response:
            raise ValueError('Formula must have response variable.')

        self._parse_rhs(rhs)

    def _parse_rhs(self, rhs: str) -> None:
        """Parse right-hand side (predictor terms).

        Handles:
            - Smooth terms: s(x), te(x1,x2), ti(x1,x2), re(group)
            - Parametric terms: x, x1*x2
            - Special: offset(var)
        """
        # Handle offset() special case
        offset_match = re.search(r'offset\s*\(\s*(\w+)\s*\)', rhs)
        if offset_match:
            self.offset_term = offset_match.group(1)
            rhs = rhs[:offset_match.start()] + rhs[offset_match.end():]

        # Split by + while respecting nested parentheses
        terms = self._split_by_plus(rhs)

        for term in terms:
            term = term.strip()
            if not term:
                continue

            # Check if smooth term
            if self._is_smooth_term(term):
                spec = self._parse_smooth_term(term)
                if spec:
                    self.smooth_terms.append(spec)
            else:
                # Parametric term
                spec = self._parse_parametric_term(term)
                if spec:
                    self.parametric_terms.append(spec)
                    self.parametric_names.extend(spec.variables)

    def _split_by_plus(self, s: str) -> list[str]:
        """Split string by '+' while respecting nested parentheses."""
        terms = []
        current = ''
        depth = 0
        for char in s:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == '+' and depth == 0:
                terms.append(current)
                current = ''
            else:
                current += char
        if current:
            terms.append(current)
        return terms

    def _is_smooth_term(self, term: str) -> bool:
        """Check if term is a smooth term (contains s, te, ti, re, etc.)."""
        return bool(self.SMOOTH_PATTERN.search(term))

    def _parse_smooth_term(self, term: str) -> SmoothSpec | None:
        """Parse a smooth term specification.

        Args:
            term: String like 's(x1)', 'te(x1, x2, k=10)', etc.

        Returns:
            SmoothSpec object or None if invalid.
        """
        match = self.SMOOTH_PATTERN.search(term)
        if not match:
            return None

        term_type = match.group(1).lower()
        arguments = match.group(2)

        # Parse arguments: variables and keyword args
        variables, kwargs = self._parse_arguments(arguments)

        if not variables:
            return None

        # Extract basis, k, m, by, fx from kwargs
        basis = kwargs.pop('basis', kwargs.pop('bs', 'tp'))
        k = kwargs.pop('k', None)
        if k is not None:
            try:
                k = int(k)
            except (ValueError, TypeError):
                k = None

        m = kwargs.pop('m', None)
        by_variable = kwargs.pop('by', None)
        fx_raw = kwargs.pop('fx', 'false')
        fx = str(fx_raw).lower() in ('true', '1', 'yes')

        # For re() terms, default basis to 're'
        if term_type == 're':
            basis = 're'
        # For t2/te/ti, basis refers to marginal basis type
        elif term_type in ('te', 'ti', 't2') and basis == 'tp':
            basis = 'tp'

        return SmoothSpec(
            term_type=term_type,
            variables=variables,
            basis=basis,
            k=k,
            m=m,
            by_variable=by_variable,
            fx=fx,
            bs_args=kwargs
        )

    def _parse_parametric_term(self, term: str) -> ParametricSpec | None:
        """Parse a parametric (linear) term.

        Args:
            term: String like 'x1', 'x1*x2', 'log(x)', etc.

        Returns:
            ParametricSpec object or None if empty.
        """
        term = term.strip()
        if not term:
            return None

        # Handle interactions
        if '*' in term:
            vars_in_interaction = [v.strip() for v in term.split('*')]
            return ParametricSpec(variables=vars_in_interaction, interaction=True)

        # Handle functions
        func_match = self.FUNCTION_PATTERN.match(term)
        if func_match:
            func_name = func_match.group(1)
            func_arg = func_match.group(2).strip()
            # Store function application
            self.function_terms[term] = f'{func_name}({func_arg})'
            return ParametricSpec(variables=[func_arg])

        # Simple variable
        if self.VARIABLE_PATTERN.fullmatch(term):
            return ParametricSpec(variables=[term])

        return None

    def _parse_arguments(self, arg_str: str) -> tuple[list[str], dict[str, Any]]:
        """Parse arguments to smooth terms.

        Returns:
            (variable_names, keyword_arguments)

        Example:
            >>> parser = FormulaParser('y ~ s(x)')
            >>> vars, kwargs = parser._parse_arguments('x1, x2, k=8, basis="cs"')
            >>> vars
            ['x1', 'x2']
            >>> kwargs
            {'k': '8', 'basis': 'cs'}
        """
        variables: list[str] = []
        kwargs: dict[str, Any] = {}

        parts = [p.strip() for p in arg_str.split(',')]
        for part in parts:
            if '=' in part:
                key, val = part.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"\'')
                kwargs[key] = val
            else:
                # It's a variable
                if part and self.VARIABLE_PATTERN.fullmatch(part):
                    variables.append(part)

        return variables, kwargs

    def summary(self) -> str:
        """Return human-readable summary of parsed formula."""
        lines = [
            f'Formula: {self.formula}',
            f'Response: {self.response}',
            f'Smooth terms ({len(self.smooth_terms)}):',
        ]
        for spec in self.smooth_terms:
            lines.append(f'  - {spec.label}')
        lines.append(f'Parametric terms ({len(self.parametric_terms)}):')
        for spec in self.parametric_terms:
            lines.append(f'  - {spec.label}')
        if self.offset_term:
            lines.append(f'Offset: {self.offset_term}')
        return '\n'.join(lines)


def parse_formula(formula: str) -> FormulaParser:
    """Parse a GAM formula into structured components.

    Args:
        formula: Formula string, e.g., 'y ~ s(x1) + s(x2, k=10) + x3'

    Returns:
        FormulaParser object with parsed components.

    Example:
        >>> parser = parse_formula('log(y) ~ s(age) + s(income) + region')
        >>> parser.response
        'log(y)'
        >>> len(parser.smooth_terms)
        2
        >>> parser.parametric_names
        ['region']
    """
    return FormulaParser(formula)
