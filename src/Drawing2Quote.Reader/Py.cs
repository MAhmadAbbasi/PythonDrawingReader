using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Drawing2Quote.Reader;

/// <summary>
/// Python / numpy behaviour the reader depends on, reproduced so the .NET reader gives the same results as the
/// Python one: rounding (half to even, on the exact binary value), float formatting, floor modulo, numpy's pairwise
/// sums, medians and percentiles, difflib's similarity ratio and Python's string helpers.
/// </summary>
internal static class Py
{
    public static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    public const RegexOptions RX = RegexOptions.CultureInvariant;

    // ── rounding ──
    /// <summary>Python round(x): half to even</summary>
    public static double Round(double x) => Math.Round(x, MidpointRounding.ToEven);

    public static int RoundInt(double x) => (int)Math.Round(x, MidpointRounding.ToEven);

    /// <summary>Python round(x, n): correctly rounded from the exact binary value, ties to even (n may be negative)</summary>
    public static double Round(double x, int n)
    {
        if (double.IsNaN(x) || double.IsInfinity(x) || x == 0) return x;
        // the exact decimal expansion, far enough past the digit to round at to tell a tie from a near tie
        var s = Math.Abs(x).ToString("F" + Math.Max(n + 40, 40), Inv);
        int dot = s.IndexOf('.');
        string digits = s.Remove(dot, 1);
        int point = dot;                                   // digits before the decimal point
        int cut = point + n;                               // digits kept
        if (cut <= 0)
        {
            // everything is below the rounding digit: 0, or 1 unit when more than half of it (ties to even 0)
            bool more = cut == 0 && (digits[0] > '5' || (digits[0] == '5' && digits.AsSpan(1).IndexOfAnyExcept('0') >= 0));
            double unit = Math.Pow(10, -n);
            return more ? Math.CopySign(unit, x) : Math.CopySign(0.0, x);
        }
        string keep = digits[..cut], rest = digits[cut..];
        int cmp = rest.Length == 0 ? -1 : rest[0] > '5' ? 1 : rest[0] < '5' ? -1 : (rest.AsSpan(1).IndexOfAnyExcept('0') >= 0 ? 1 : 0);
        var sb = new StringBuilder(keep);
        bool up = cmp > 0 || (cmp == 0 && (sb[^1] - '0') % 2 == 1);
        if (up)
        {
            int i = sb.Length - 1;
            while (i >= 0)
            {
                if (sb[i] == '9') { sb[i] = '0'; i--; }
                else { sb[i]++; break; }
            }
            if (i < 0) { sb.Insert(0, '1'); point++; }
        }
        var str = sb.ToString();
        string num = n > 0 ? str[..(str.Length - n)] + "." + str[(str.Length - n)..] : str + new string('0', -n);
        var v = double.Parse(num, Inv);
        return x < 0 ? -v : v;
    }

    /// <summary>Python / numpy float a % b (the result takes the sign of b)</summary>
    public static double Mod(double a, double b)
    {
        double m = a % b;                                  // C fmod
        if (m != 0 && (b < 0) != (m < 0)) m += b;
        return m;
    }

    /// <summary>Python a // b for floats</summary>
    public static double FloorDiv(double a, double b)
    {
        double mod = a % b;
        double div = (a - mod) / b;
        if (mod != 0 && (b < 0) != (mod < 0)) div -= 1.0;
        if (div == 0) return Math.CopySign(0.0, a / b);
        double floordiv = Math.Floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
        return floordiv;
    }

    // ── formatting ──
    /// <summary>the shortest digits that read back as x, and the power of ten of the first one</summary>
    private static (string digits, int exp) Shortest(double x)
    {
        var r = Math.Abs(x).ToString("R", Inv);
        int e = 0;
        int ei = r.IndexOfAny(new[] { 'E', 'e' });
        if (ei >= 0) { e = int.Parse(r[(ei + 1)..], Inv); r = r[..ei]; }
        int dot = r.IndexOf('.');
        int intLen = dot < 0 ? r.Length : dot;
        string digits = r.Replace(".", "");
        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0') lead++;
        digits = digits[lead..].TrimEnd('0');
        if (digits.Length == 0) digits = "0";
        return (digits, intLen - lead - 1 + e);
    }

    /// <summary>Python str(float) / repr(float): shortest round trip, "1.0" for whole numbers</summary>
    public static string Repr(double x)
    {
        if (double.IsNaN(x)) return "nan";
        if (double.IsInfinity(x)) return x > 0 ? "inf" : "-inf";
        string sign = double.IsNegative(x) ? "-" : "";
        if (x == 0) return sign + "0.0";
        var (d, e) = Shortest(x);
        if (e < -4 || e >= 16)
            return sign + d[0] + (d.Length > 1 ? "." + d[1..] : "") + "e" + (e < 0 ? "-" : "+") + Math.Abs(e).ToString("00", Inv);
        if (e < 0) return sign + "0." + new string('0', -e - 1) + d;
        if (d.Length <= e + 1) return sign + d + new string('0', e + 1 - d.Length) + ".0";
        return sign + d[..(e + 1)] + "." + d[(e + 1)..];
    }

    /// <summary>Python format(x, "g") (6 significant digits, trailing zeros dropped)</summary>
    public static string G(double x, int precision = 6)
    {
        if (double.IsNaN(x)) return "nan";
        if (double.IsInfinity(x)) return x > 0 ? "inf" : "-inf";
        if (x == 0) return double.IsNegative(x) ? "-0" : "0";
        double a = Math.Abs(x);
        int e = (int)Math.Floor(Math.Log10(a));
        if (a < Math.Pow(10, e)) e--;
        if (a >= Math.Pow(10, e + 1)) e++;
        double r = Round(a, precision - 1 - e);
        if (r >= Math.Pow(10, e + 1)) e++;
        string sign = x < 0 ? "-" : "";
        if (e < -4 || e >= precision)
        {
            var m = Fixed(r / Math.Pow(10, e), precision - 1);
            if (m.Contains('.')) m = m.TrimEnd('0').TrimEnd('.');
            return sign + m + "e" + (e < 0 ? "-" : "+") + Math.Abs(e).ToString("00", Inv);
        }
        var f = Fixed(r, Math.Max(precision - 1 - e, 0));
        if (f.Contains('.')) f = f.TrimEnd('0').TrimEnd('.');
        return sign + f;
    }

    private static readonly double[] Pow10 = { 1, 10, 100, 1000, 10000, 100000, 1000000 };

    /// <summary>Python f"{x:.nf}" for n up to 6, without going through a long decimal expansion (the content
    /// streams of traced drawings have millions of numbers): x * 10^n with its exact rounding error (FMA) decides</summary>
    public static string FixedFast(double x, int decimals)
    {
        double scale = Pow10[decimals];
        double a = Math.Abs(x);
        double t = a * scale;
        if (!(t < 4e15)) return Fixed(x, decimals);
        double err = Math.FusedMultiplyAdd(a, scale, -t);    // a * scale = t + err exactly
        double fl = Math.Floor(t);
        double frac = t - fl;
        // (frac is a multiple of t's last bit and err below half of it: only an exact half needs err)
        bool up = frac > 0.5 || (frac == 0.5 && (err > 0 || (err == 0 && ((long)fl & 1) == 1)));
        long units = (long)fl + (up ? 1 : 0);
        var digits = units.ToString(Inv);
        if (decimals > 0)
        {
            if (digits.Length <= decimals) digits = new string('0', decimals - digits.Length + 1) + digits;
            digits = digits[..^decimals] + "." + digits[^decimals..];
        }
        return (double.IsNegative(x) ? "-" : "") + digits;
    }

    /// <summary>Python f"{x:.nf}": the exact value rounded half to even</summary>
    public static string Fixed(double x, int decimals) => Round(x, decimals).ToString("F" + decimals, Inv);

    // ── statistics / numpy ──
    /// <summary>statistics.median / numpy.median of doubles</summary>
    public static double Median(IEnumerable<double> values)
    {
        var a = values.ToArray();
        if (a.Length == 0) throw new InvalidOperationException("no median for empty data");
        Array.Sort(a);
        int n = a.Length;
        return n % 2 == 1 ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
    }

    /// <summary>numpy.median of a float32 array (the mean of the two middle values in float32)</summary>
    public static float MedianF(IEnumerable<float> values)
    {
        var a = values.ToArray();
        Array.Sort(a);
        int n = a.Length;
        return n % 2 == 1 ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2f;
    }

    /// <summary>numpy's pairwise summation (np.sum of a 1-d float64 array)</summary>
    public static double NpSum(ReadOnlySpan<double> a)
    {
        int n = a.Length;
        if (n < 8)
        {
            double res = 0.0;
            for (int i = 0; i < n; i++) res += a[i];
            return res;
        }
        if (n <= 128)
        {
            double r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
            int i;
            for (i = 8; i < n - (n % 8); i += 8)
            {
                r0 += a[i]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3];
                r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7];
            }
            double res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) res += a[i];
            return res;
        }
        int n2 = n / 2;
        n2 -= n2 % 8;
        return NpSum(a[..n2]) + NpSum(a[n2..]);
    }

    public static double NpSum(IEnumerable<double> a) => NpSum(a.ToArray());

    /// <summary>numpy.percentile(a, q) with the default linear method</summary>
    public static double Percentile(double[] values, double q)
    {
        var a = (double[])values.Clone();
        Array.Sort(a);
        int n = a.Length;
        double quantile = q / 100.0;
        double virt = (n - 1) * quantile;
        int prev = (int)Math.Floor(virt);
        int next = prev + 1;
        if (prev < 0) prev = 0;
        if (prev > n - 1) prev = n - 1;
        if (next > n - 1) next = n - 1;
        double gamma = virt - Math.Floor(virt);
        double lo = a[prev], hi = a[next];
        double diff = hi - lo;
        return gamma >= 0.5 ? hi - diff * (1 - gamma) : lo + diff * gamma;
    }

    // ── difflib.SequenceMatcher(None, a, b).ratio() ──
    public static double SequenceRatio(string a, string b)
    {
        int la = a.Length, lb = b.Length;
        if (la + lb == 0) return 1.0;
        // b2j: positions of each character in b; popular characters purged (autojunk) as difflib does
        var b2j = new Dictionary<char, List<int>>();
        for (int i = 0; i < lb; i++)
        {
            if (!b2j.TryGetValue(b[i], out var l)) b2j[b[i]] = l = new List<int>();
            l.Add(i);
        }
        if (lb >= 200)
        {
            int ntest = lb / 100 + 1;
            foreach (var key in b2j.Where(kv => kv.Value.Count > ntest).Select(kv => kv.Key).ToList()) b2j.Remove(key);
        }
        int matches = 0;
        var queue = new Stack<(int, int, int, int)>();
        queue.Push((0, la, 0, lb));
        while (queue.Count > 0)
        {
            var (alo, ahi, blo, bhi) = queue.Pop();
            var (i, j, k) = LongestMatch(a, b, b2j, alo, ahi, blo, bhi);
            if (k > 0)
            {
                matches += k;
                if (alo < i && blo < j) queue.Push((alo, i, blo, j));
                if (i + k < ahi && j + k < bhi) queue.Push((i + k, ahi, j + k, bhi));
            }
        }
        return 2.0 * matches / (la + lb);
    }

    private static (int, int, int) LongestMatch(string a, string b, Dictionary<char, List<int>> b2j, int alo, int ahi, int blo, int bhi)
    {
        int besti = alo, bestj = blo, bestsize = 0;
        var j2len = new Dictionary<int, int>();
        for (int i = alo; i < ahi; i++)
        {
            var newj2len = new Dictionary<int, int>();
            if (b2j.TryGetValue(a[i], out var js))
            {
                foreach (int j in js)
                {
                    if (j < blo) continue;
                    if (j >= bhi) break;
                    int k = (j2len.TryGetValue(j - 1, out var v) ? v : 0) + 1;
                    newj2len[j] = k;
                    if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k; }
                }
            }
            j2len = newj2len;
        }
        // (no junk: popular characters are only left out of b2j, so they extend matches here as in difflib)
        while (besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1])
        {
            besti--; bestj--; bestsize++;
        }
        while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize])
            bestsize++;
        return (besti, bestj, bestsize);
    }

    // ── strings ──
    /// <summary>Python str.split() with no separator</summary>
    public static string[] Split(string s) => s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

    /// <summary>Python str.splitlines()</summary>
    public static List<string> SplitLines(string s)
    {
        var out_ = new List<string>();
        int start = 0, i = 0;
        while (i < s.Length)
        {
            char c = s[i];
            if (c is '\n' or '\r' or '\v' or '\f' or '\x1c' or '\x1d' or '\x1e' or '\x85' or (char)0x2028 or (char)0x2029)
            {
                out_.Add(s[start..i]);
                if (c == '\r' && i + 1 < s.Length && s[i + 1] == '\n') i++;
                start = i + 1;
            }
            i++;
        }
        if (start < s.Length) out_.Add(s[start..]);
        return out_;
    }

    public static bool IsAlpha(string s) => s.Length > 0 && s.All(char.IsLetter);

    public static bool IsDigit(string s) => s.Length > 0 && s.All(char.IsDigit);

    /// <summary>Python str.capitalize()</summary>
    public static string Capitalize(string s) => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant();

    /// <summary>Python str.title() for a single word</summary>
    public static string Title(string s)
    {
        var sb = new StringBuilder(s.Length);
        bool prevLetter = false;
        foreach (var c in s)
        {
            sb.Append(prevLetter ? char.ToLowerInvariant(c) : char.ToUpperInvariant(c));
            prevLetter = char.IsLetter(c);
        }
        return sb.ToString();
    }

    /// <summary>float(text) for digits matched by \d (which can be other scripts' digits, as in Python)</summary>
    public static double ParseFloat(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var c in s)
            sb.Append(char.IsDigit(c) && (c < '0' || c > '9') ? (char)('0' + (int)char.GetNumericValue(c)) : c);
        return double.Parse(sb.ToString(), NumberStyles.Float, Inv);
    }

    public static int ParseInt(string s) => (int)ParseFloat(s);

    // ── regular expressions ──
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, Regex> Cache = new();

    public static Regex Rx(string pattern) => Cache.GetOrAdd(pattern, p => new Regex(p, RX));

    /// <summary>re.search(pattern, s)</summary>
    public static bool Search(string pattern, string s) => Rx(pattern).IsMatch(s);

    /// <summary>re.match(pattern, s): a match starting at the beginning</summary>
    public static Match? MatchStart(string pattern, string s)
    {
        var m = Rx(@"\A(?:" + pattern + ")").Match(s);
        return m.Success ? m : null;
    }

    /// <summary>re.fullmatch(pattern, s)</summary>
    public static Match? FullMatch(string pattern, string s)
    {
        var m = Rx(@"\A(?:" + pattern + @")\z").Match(s);
        return m.Success ? m : null;
    }

    /// <summary>min(items, key=...) in Python: the first item with the smallest key</summary>
    public static T MinBy<T, K>(IEnumerable<T> items, Func<T, K> key) where K : IComparable<K>
    {
        bool any = false;
        T best = default!;
        K bestKey = default!;
        foreach (var it in items)
        {
            var k = key(it);
            if (!any || k.CompareTo(bestKey) < 0) { best = it; bestKey = k; any = true; }
        }
        if (!any) throw new InvalidOperationException("min() arg is an empty sequence");
        return best;
    }

    /// <summary>max(items, key=...) in Python: the first item with the largest key</summary>
    public static T MaxBy<T, K>(IEnumerable<T> items, Func<T, K> key) where K : IComparable<K>
    {
        bool any = false;
        T best = default!;
        K bestKey = default!;
        foreach (var it in items)
        {
            var k = key(it);
            if (!any || k.CompareTo(bestKey) > 0) { best = it; bestKey = k; any = true; }
        }
        if (!any) throw new InvalidOperationException("max() arg is an empty sequence");
        return best;
    }

    /// <summary>Python's stable sort (List.Sort is not stable)</summary>
    public static List<T> Sorted<T, K>(IEnumerable<T> items, Func<T, K> key, bool reverse = false)
    {
        var list = items.Select((x, i) => (x, i)).ToList();
        var cmp = Comparer<K>.Default;
        list.Sort((p, q) =>
        {
            int c = cmp.Compare(key(p.x), key(q.x));
            if (reverse) c = -c;
            return c != 0 ? c : p.i.CompareTo(q.i);
        });
        return list.Select(p => p.x).ToList();
    }
}

/// <summary>Tuples compared item by item, as Python compares tuples (for sort keys)</summary>
internal readonly record struct Key2(double A, double B) : IComparable<Key2>
{
    public int CompareTo(Key2 o) { int c = A.CompareTo(o.A); return c != 0 ? c : B.CompareTo(o.B); }
}

internal readonly record struct Key3(double A, double B, double C) : IComparable<Key3>
{
    public int CompareTo(Key3 o)
    {
        int c = A.CompareTo(o.A);
        if (c != 0) return c;
        c = B.CompareTo(o.B);
        return c != 0 ? c : C.CompareTo(o.C);
    }
}
