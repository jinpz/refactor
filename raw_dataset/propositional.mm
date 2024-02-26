$(
###############################################################################
            CLASSICAL FIRST-ORDER LOGIC WITH EQUALITY
###############################################################################

  Logic can be defined as the "study of the principles of correct reasoning"
  (Merrilee H. Salmon's 1991 "Informal Reasoning and Informal Logic" in
  _Informal Reasoning and Education_) or as "a formal system using symbolic
  techniques and mathematical methods to establish truth-values" (the Oxford
  English Dictionary).

  This section formally defines the logic system we will use.  In particular,
  it defines symbols for declaring truthful statements, along with rules for
  deriving truthful statements from other truthful statements.  The system
  defined here is classical first-order logic with equality (the most common
  logic system used by mathematicians).

  We begin with a few housekeeping items in pre-logic, and then introduce
  propositional calculus (both its axioms and important theorems that can be
  derived from them).  Propositional calculus deals with general truths about
  well-formed formulas (wffs) regardless of how they are constructed.  This is
  followed by proofs that other axiomatizations of classical propositional
  calculus can be derived from the axioms we have chosen to use.

  We then define predicate calculus, which adds additional symbols and rules
  useful for discussing objects (beyond simply true or false).  In particular,
  it introduces the symbols ` = ` ("equals"), ` e. ` ("is a member of"), and `
  A. ` ("for all").  The first two are called "predicates."  A predicate
  specifies a true or false relationship between its two arguments.

$)

$(
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
                           Pre-logic
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

  This section includes a few "housekeeping" mechanisms before we begin
  defining the basics of logic.

$)

  $( Declare the primitive constant symbols for propositional calculus. $)
  $c ( $.  $( Left parenthesis $)
  $c ) $.  $( Right parenthesis $)
  $c -> $. $( Right arrow (read:  "implies") $)
  $c -. $. $( Right handle (read:  "not") $)
  $c wff $. $( Well-formed formula symbol (read:  "the following symbol
               sequence is a wff") $)
  $c |- $. $( Turnstile (read:  "the following symbol sequence is provable" or
              'a proof exists for") $)

  $( Define the syntax and logical typecodes, and declare that our grammar is
     unambiguous (verifiable using the KLR parser, with compositing depth 5).
     (This $ j comment need not be read by verifiers, but is useful for parsers
     like mmj2.) $)
  $( $j
    syntax 'wff';
    syntax '|-' as 'wff';
    unambiguous 'klr 5';
  $)

  $( Declare typographical constant symbols that are not directly used
     in the formalism, but *are* symbols we find useful when
     explaining the formalism. It is much easier to consistently use
     a single typographical system when generating text. $)

  $c & $. $( Ampersand (read: "and-also") $)
  $c => $. $( Big-to (read: "proves") $)

  $( wff variable sequence:  ph ps ch th ta et ze si rh mu la ka $)
  $( Introduce some variable names we will use to represent well-formed
     formulas (wff's). $)
  $v ph $.  $( Greek phi $)
  $v ps $.  $( Greek psi $)
  $v ch $.  $( Greek chi $)
  $v th $.  $( Greek theta $)
  $v ta $.  $( Greek tau $)
  $v et $.  $( Greek eta $)
  $v ze $.  $( Greek zeta $)
  $v si $.  $( Greek sigma $)
  $v rh $.  $( Greek rho $)
  $v mu $.  $( Greek mu $)
  $v la $.  $( Greek lambda $)
  $v ka $.  $( Greek kappa $)

  $( Specify some variables that we will use to represent wff's.
     The fact that a variable represents a wff is relevant only to a theorem
     referring to that variable, so we may use $f hypotheses.  The symbol
     ` wff ` specifies that the variable that follows it represents a wff. $)
  $( Let variable ` ph ` be a wff. $)
  wph $f wff ph $.
  $( Let variable ` ps ` be a wff. $)
  wps $f wff ps $.
  $( Let variable ` ch ` be a wff. $)
  wch $f wff ch $.
  $( Let variable ` th ` be a wff. $)
  wth $f wff th $.
  $( Let variable ` ta ` be a wff. $)
  wta $f wff ta $.
  $( Let variable ` et ` be a wff. $)
  wet $f wff et $.
  $( Let variable ` ze ` be a wff. $)
  wze $f wff ze $.
  $( Let variable ` si ` be a wff. $)
  wsi $f wff si $.
  $( Let variable ` rh ` be a wff. $)
  wrh $f wff rh $.
  $( Let variable ` mu ` be a wff. $)
  wmu $f wff mu $.
  $( Let variable ` la ` be a wff. $)
  wla $f wff la $.
  $( Let variable ` ka ` be a wff. $)
  wka $f wff ka $.

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
                 Inferences for assisting proof development
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
$)

  ${
    dummylink.1 $e |- ph $.
    dummylink.2 $e |- ps $.
    $( (_Note_:  This inference rule and the next one, ~ idi , will normally
       never appear in a completed proof.  They can be ignored if you are using
       this database to assist learning logic - please start with the statement
       ~ wn instead.)

       This is a technical inference to assist proof development.  It provides
       a temporary way to add an independent subproof to a proof under
       development, for later assignment to a normal proof step.

       The metamath program's Proof Assistant requires proofs to be developed
       backwards from the conclusion with no gaps, and it has no mechanism that
       lets the user to work on isolated subproofs.  This inference provides a
       workaround for this limitation.  It can be inserted at any point in a
       proof to allow an independent subproof to be developed on the side, for
       later use as part of the final proof.

       _Instructions_:  (1) Assign this inference to any unknown step in the
       proof.  Typically, the last unknown step is the most convenient, since
       'assign last' can be used.  This step will be replicated in hypothesis
       dummylink.1, from where the development of the main proof can continue.
       (2) Develop the independent subproof backwards from hypothesis
       dummylink.2.  If desired, use a 'let' command to pre-assign the
       conclusion of the independent subproof to dummylink.2.  (3) After the
       independent subproof is complete, use 'improve all' to assign it
       automatically to an unknown step in the main proof that matches it.  (4)
       After the entire proof is complete, use 'minimize *' to clean up
       (discard) all dummylink references automatically.

       This inference was originally designed to assist importing partially
       completed Proof Worksheets from the mmj2 Proof Assistant GUI, but it can
       also be useful on its own.  Interestingly, no axioms are required for
       its proof.  It is the inference associated with ~ a1i .  (Contributed by
       NM, 7-Feb-2006.) $)
    dummylink $p |- ph $=
      (  ) C $.
  $}

  ${
    idi.1 $e |- ph $.
    $( Inference form of ~ id .  This inference rule, which requires no axioms
       for its proof, is useful as a copy-paste mechanism during proof
       development in mmj2.  It is normally not referenced in the final version
       of a proof, since it is always redundant and can be removed using the
       'minimize *' command in the metamath program's Proof Assistant.
       Contrary to its closed form ~ id , it requires no axioms for its proof.
       (Contributed by Alan Sare, 31-Dec-2011.) $)
    idi $p |- ph $=
      (  ) B $.
  $}

$(
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
                           Propositional calculus
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

  Propositional calculus deals with general truths about well-formed formulas
  (wffs) regardless of how they are constructed.  The simplest propositional
  truth is ` ( ph -> ph ) ` , which can be read "if something is true, then it
  is true" - rather trivial and obvious, but nonetheless it must be proved from
  the axioms (see theorem ~ id ).

  Our system of propositional calculus consists of three basic axioms and
  another axiom that defines the modus-ponens inference rule.  It is attributed
  to Jan Lukasiewicz (pronounced woo-kah-SHAY-vitch) and was popularized by
  Alonzo Church, who called it system P2.  (Thanks to Ted Ulrich for this
  information.)  These axioms are ~ ax-1 , ~ ax-2 , ~ ax-3 , and (for modus
  ponens) ~ ax-mp . Some closely followed texts include [Margaris] for the
  axioms and [WhiteheadRussell] for the theorems.

  The propositional calculus used here is the classical system widely used by
  mathematicians.  In particular, this logic system accepts the "law of the
  excluded middle" as proven in ~ exmid , which says that a logical statement
  is either true or not true.  This is an essential distinction of classical
  logic and is not a theorem of intuitionistic logic.

  All 194 axioms, definitions, and theorems for propositional calculus in
  _Principia Mathematica_ (specifically *1.2 through *5.75) are axioms or
  formally proven.  See the Bibliographic Cross-References at
  ~ http://us.metamath.org/mpeuni/mmbiblio.html for a complete
  cross-reference from sources used to its formalization in the Metamath
  Proof Explorer.

$)

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Recursively define primitive wffs for propositional calculus
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
$)

  $( If ` ph ` is a wff, so is ` -. ph ` or "not ` ph ` ."  Part of the
     recursive definition of a wff (well-formed formula).  In classical logic
     (which is our logic), a wff is interpreted as either true or false.  So if
     ` ph ` is true, then ` -. ph ` is false; if ` ph ` is false, then
     ` -. ph ` is true.  Traditionally, Greek letters are used to represent
     wffs, and we follow this convention.  In propositional calculus, we define
     only wffs built up from other wffs, i.e. there is no starting or "atomic"
     wff.  Later, in predicate calculus, we will extend the basic wff
     definition by including atomic wffs ( ~ weq and ~ wel ). $)
  wn $a wff -. ph $.

  $( If ` ph ` and ` ps ` are wff's, so is ` ( ph -> ps ) ` or " ` ph ` implies
     ` ps ` ."  Part of the recursive definition of a wff.  The resulting wff
     is (interpreted as) false when ` ph ` is true and ` ps ` is false; it is
     true otherwise.  Think of the truth table for an OR gate with input ` ph `
     connected through an inverter.  After we define the axioms of
     propositional calculus ( ~ ax-1 , ~ ax-2 , ~ ax-3 , and ~ ax-mp ), the
     biconditional ( ~ df-bi ), the constant true ` T. ` ( ~ df-tru ), and the
     constant false ` F. ` ( ~ df-fal ), we will be able to prove these truth
     table values: ` ( ( T. -> T. ) <-> T. ) ` ( ~ truimtru ),
     ` ( ( T. -> F. ) <-> F. ) ` ( ~ truimfal ), ` ( ( F. -> T. ) <-> T. ) `
     ( ~ falimtru ), and ` ( ( F. -> F. ) <-> T. ) ` ( ~ falimfal ).  These
     have straightforward meanings, for example, ` ( ( T. -> T. ) <-> T. ) `
     just means "the value of ` ( T. -> T. ) ` is ` T. ` ".

     The left-hand wff is called the antecedent, and the right-hand wff is
     called the consequent.  In the case of ` ( ph -> ( ps -> ch ) ) ` , the
     middle ` ps ` may be informally called either an antecedent or part of the
     consequent depending on context.  Contrast with ` <-> ` ( ~ df-bi ),
     ` /\ ` ( ~ df-an ), and ` \/ ` ( ~ df-or ).

     This is called "material implication" and the arrow is usually read as
     "implies."  However, material implication is not identical to the meaning
     of "implies" in natural language.  For example, the word "implies" may
     suggest a causal relationship in natural language.  Material implication
     does not require any causal relationship.  Also, note that in material
     implication, if the consequent is true then the wff is always true (even
     if the antecedent is false).  Thus, if "implies" means material
     implication, it is true that "if the moon is made of green cheese that
     implies that 5=5" (because 5=5).  Similarly, if the antecedent is false,
     the wff is always true.  Thus, it is true that, "if the moon made of green
     cheese that implies that 5=7" (because the moon is not actually made of
     green cheese).  A contradiction implies anything ( ~ pm2.21i ).  In short,
     material implication has a very specific technical definition, and
     misunderstandings of it are sometimes called "paradoxes of logical
     implication." $)
  wi $a wff ( ph -> ps ) $.

  $( Register '-.' and '->' as primitive expressions (lacking definitions). $)
  $( $j primitive 'wn' 'wi'; $)

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        The axioms of propositional calculus
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  Propositional calculus (axioms ~ ax-1 through ~ ax-3 and rule ~ ax-mp ) can
  be thought of as asserting formulas that are universally "true" when their
  variables are replaced by any combination of "true" and "false."
  Propositional calculus was first formalized by Frege in 1879, using as his
  axioms (in addition to rule ~ ax-mp ) the wffs ~ ax-1 , ~ ax-2 , ~ pm2.04 ,
  ~ con3 , ~ notnot2 , and ~ notnot1 . Around 1930, Lukasiewicz simplified the
  system by eliminating the third (which follows from the first two, as you can
  see by looking at the proof of ~ pm2.04 ) and replacing the last three with
  our ~ ax-3 . (Thanks to Ted Ulrich for this information.)

  The theorems of propositional calculus are also called _tautologies_.
  Tautologies can be proved very simply using truth tables, based on the
  true/false interpretation of propositional calculus.  To do this, we assign
  all possible combinations of true and false to the wff variables and verify
  that the result (using the rules described in ~ wi and ~ wn ) always
  evaluates to true.  This is called the _semantic_ approach.  Our approach is
  called the _syntactic_ approach, in which everything is derived from axioms.
  A metatheorem called the Completeness Theorem for Propositional Calculus
  shows that the two approaches are equivalent and even provides an algorithm
  for automatically generating syntactic proofs from a truth table.  Those
  proofs, however, tend to be long, since truth tables grow exponentially with
  the number of variables, and the much shorter proofs that we show here were
  found manually.

$)

  ${
    $( Minor premise for modus ponens. $)
    min $e |- ph $.
    $( Major premise for modus ponens. $)
    maj $e |- ( ph -> ps ) $.
    $( Rule of Modus Ponens.  The postulated inference rule of propositional
       calculus.  See e.g.  Rule 1 of [Hamilton] p. 73.  The rule says, "if
       ` ph ` is true, and ` ph ` implies ` ps ` , then ` ps ` must also be
       true."  This rule is sometimes called "detachment," since it detaches
       the minor premise from the major premise.  "Modus ponens" is short for
       "modus ponendo ponens," a Latin phrase that means "the mode that by
       affirming affirms" - remark in [Sanford] p. 39.  This rule is similar to
       the rule of modus tollens ~ mto .

       Note:  In some web page displays such as the Statement List, the
       symbols " ` & ` " and " ` => ` " informally indicate the relationship
       between the hypotheses and the assertion (conclusion), abbreviating the
       English words "and" and "implies."  They are not part of the formal
       language.  (Contributed by NM, 30-Sep-1992.) $)
    ax-mp $a |- ps $.
  $}

  $( Axiom _Simp_.  Axiom A1 of [Margaris] p. 49.  One of the 3 axioms of
     propositional calculus.  The 3 axioms are also given as Definition 2.1 of
     [Hamilton] p. 28.  This axiom is called _Simp_ or "the principle of
     simplification" in _Principia Mathematica_ (Theorem *2.02 of
     [WhiteheadRussell] p. 100) because "it enables us to pass from the joint
     assertion of ` ph ` and ` ps ` to the assertion of ` ph ` simply."  It is
     Proposition 1 of [Frege1879] p. 26, its first axiom.  (Contributed by NM,
     30-Sep-1992.) $)
  ax-1 $a |- ( ph -> ( ps -> ph ) ) $.

  $( Axiom _Frege_.  Axiom A2 of [Margaris] p. 49.  One of the 3 axioms of
     propositional calculus.  It "distributes" an antecedent over two
     consequents.  This axiom was part of Frege's original system and is known
     as _Frege_ in the literature; see Proposition 2 of [Frege1879] p. 26.  It
     is also proved as Theorem *2.77 of [WhiteheadRussell] p. 108.  The other
     direction of this axiom also turns out to be true, as demonstrated by
     ~ pm5.41 .  (Contributed by NM, 30-Sep-1992.) $)
  ax-2 $a |- ( ( ph -> ( ps -> ch ) ) -> ( ( ph -> ps ) -> ( ph -> ch ) ) ) $.

  $( Axiom _Transp_.  Axiom A3 of [Margaris] p. 49.  One of the 3 axioms of
     propositional calculus.  It swaps or "transposes" the order of the
     consequents when negation is removed.  An informal example is that the
     statement "if there are no clouds in the sky, it is not raining" implies
     the statement "if it is raining, there are clouds in the sky."  This axiom
     is called _Transp_ or "the principle of transposition" in _Principia
     Mathematica_ (Theorem *2.17 of [WhiteheadRussell] p. 103).  We will also
     use the term "contraposition" for this principle, although the reader is
     advised that in the field of philosophical logic, "contraposition" has a
     different technical meaning.  (Contributed by NM, 30-Sep-1992.) $)
  ax-3 $a |- ( ( -. ph -> -. ps ) -> ( ps -> ph ) ) $.

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Logical implication
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  The results in this section are based on implication only, and avoid
  ~ ax-3 , so are intuitionistic.  In an implication, the wff before the
  arrow is called the "antecedent" and the wff after the arrow is called
  the "consequent."

$)

  ${
    mp2.1 $e |- ph $.
    mp2.2 $e |- ps $.
    mp2.3 $e |- ( ph -> ( ps -> ch ) ) $.
    $( A double modus ponens inference.  (Contributed by NM, 5-Apr-1994.)
       (Proof modification is discouraged.) $)
    mp2 $p |- ch $=
      ( wi ax-mp ) BCEABCGDFHH $.
  $}

  ${
    mp2b.1 $e |- ph $.
    mp2b.2 $e |- ( ph -> ps ) $.
    mp2b.3 $e |- ( ps -> ch ) $.
    $( A double modus ponens inference.  (Contributed by Mario Carneiro,
       24-Jan-2013.) $)
    mp2b $p |- ch $=
      ( ax-mp ) BCABDEGFG $.
  $}

  ${
    $( Premise for ~ a1i . $)
    a1i.1 $e |- ph $.
    $( Inference derived from axiom ~ ax-1 .  See ~ a1d for an explanation of
       our informal use of the terms "inference" and "deduction."  See also the
       comment in ~ syld .  (Contributed by NM, 29-Dec-1992.) $)
    a1i $p |- ( ps -> ph ) $=
      ( wi ax-1 ax-mp ) ABADCABEF $.
  $}

  ${
    mp1i.a $e |- ph $.
    mp1i.b $e |- ( ph -> ps ) $.
    $( Drop and replace an antecedent.  (Contributed by Stefan O'Rear,
       29-Jan-2015.) $)
    mp1i $p |- ( ch -> ps ) $=
      ( ax-mp a1i ) BCABDEFG $.
  $}

  ${
    $( Premise for ~ a2i . $)
    a2i.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Inference derived from axiom ~ ax-2 .  (Contributed by NM,
       29-Dec-1992.) $)
    a2i $p |- ( ( ph -> ps ) -> ( ph -> ch ) ) $=
      ( wi ax-2 ax-mp ) ABCEEABEACEEDABCFG $.
  $}

  ${
    imim2i.1 $e |- ( ph -> ps ) $.
    $( Inference adding common antecedents in an implication.  (Contributed by
       NM, 28-Dec-1992.) $)
    imim2i $p |- ( ( ch -> ph ) -> ( ch -> ps ) ) $=
      ( wi a1i a2i ) CABABECDFG $.
  $}

  ${
    mpd.1 $e |- ( ph -> ps ) $.
    mpd.2 $e |- ( ph -> ( ps -> ch ) ) $.
    $( A modus ponens deduction.  A translation of natural deduction rule
       ` -> ` E ( ` -> ` elimination), see ~ natded .  Deduction form of
       ~ ax-mp and inference associated with ~ a2i .  (Contributed by NM,
       29-Dec-1992.) $)
    mpd $p |- ( ph -> ch ) $=
      ( wi a2i ax-mp ) ABFACFDABCEGH $.
  $}

  ${
    $( First of 2 premises for ~ syl . $)
    syl.1 $e |- ( ph -> ps ) $.
    $( Second of 2 premises for ~ syl . $)
    syl.2 $e |- ( ps -> ch ) $.
    $( An inference version of the transitive laws for implication ~ imim2 and
       ~ imim1 (and ~ imim1i and ~ imim2i ), which Russell and Whitehead call
       "the principle of the syllogism...because...the syllogism in Barbara is
       derived from them" (quote after Theorem *2.06 of [WhiteheadRussell]
       p. 101).  Some authors call this law a "hypothetical syllogism."

       (A bit of trivia: this is the most commonly referenced assertion in our
       database.  In second place is ~ eqid , followed by ~ syl2anc ,
       ~ adantr , ~ syl3anc , and ~ ax-mp .  The Metamath program command 'show
       usage' shows the number of references.)  (Contributed by NM,
       30-Sep-1992.)  (Proof shortened by Mel L. O'Cat, 20-Oct-2011.)  (Proof
       shortened by Wolf Lammen, 26-Jul-2012.) $)
    syl $p |- ( ph -> ch ) $=
      ( wi a1i mpd ) ABCDBCFAEGH $.
  $}

  ${
    mpi.1 $e |- ps $.
    mpi.2 $e |- ( ph -> ( ps -> ch ) ) $.
    $( A nested modus ponens inference.  Inference associated with ~ com12 .
       (Contributed by NM, 29-Dec-1992.)  (Proof shortened by Stefan Allan,
       20-Mar-2006.) $)
    mpi $p |- ( ph -> ch ) $=
      ( a1i mpd ) ABCBADFEG $.
  $}

  ${
    mpisyl.1 $e |- ( ph -> ps ) $.
    mpisyl.2 $e |- ch $.
    mpisyl.3 $e |- ( ps -> ( ch -> th ) ) $.
    $( A syllogism combined with a modus ponens inference.  (Contributed by
       Alan Sare, 25-Jul-2011.) $)
    mpisyl $p |- ( ph -> th ) $=
      ( mpi syl ) ABDEBCDFGHI $.
  $}

  ${
    mp2OLD.1 $e |- ph $.
    mp2OLD.2 $e |- ps $.
    mp2OLD.3 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Obsolete proof of ~ mp2 as of 21-Jul-2019.  Shorter proof, but uses two
       more axioms.  (Contributed by Wolf Lammen, 23-Jul-2013.)
       (New usage is discouraged.)  (Proof modification is discouraged.) $)
    mp2OLD $p |- ch $=
      ( mpi ax-mp ) ACDABCEFGH $.
  $}

  ${
    3syl.1 $e |- ( ph -> ps ) $.
    3syl.2 $e |- ( ps -> ch ) $.
    3syl.3 $e |- ( ch -> th ) $.
    $( Inference chaining two syllogisms ~ syl .  (Contributed by NM,
       28-Dec-1992.) $)
    3syl $p |- ( ph -> th ) $=
      ( syl ) ACDABCEFHGH $.
  $}

  ${
    4syl.1 $e |- ( ph -> ps ) $.
    4syl.2 $e |- ( ps -> ch ) $.
    4syl.3 $e |- ( ch -> th ) $.
    4syl.4 $e |- ( th -> ta ) $.
    $( Inference chaining three syllogisms ~ syl .  (Contributed by BJ,
       14-Jul-2018.)  The use of this theorem is marked "discouraged" because
       it can cause the "minimize" command to have very long run times.
       However, feel free to use "minimize 4syl /override" if you wish.
       Remember to update the Travis "discouraged" file if it gets used.
       (New usage is discouraged.) $)
    4syl $p |- ( ph -> ta ) $=
      ( 3syl syl ) ADEABCDFGHJIK $.
  $}

  $( Principle of identity.  Theorem *2.08 of [WhiteheadRussell] p. 101.  For
     another version of the proof directly from axioms, see ~ id1 .  Its
     associated inference, ~ idi , requires no axioms for its proof, contrary
     to ~ id .  (Contributed by NM, 29-Dec-1992.)  (Proof shortened by Stefan
     Allan, 20-Mar-2006.) $)
  id $p |- ( ph -> ph ) $=
    ( wi ax-1 mpd ) AAABZAAACAECD $.

  $( Alternate proof of ~ id .  This version is proved directly from the axioms
     for demonstration purposes.  This proof is a popular example in the
     literature and is identical, step for step, to the proofs of Theorem 1 of
     [Margaris] p. 51, Example 2.7(a) of [Hamilton] p. 31, Lemma 10.3 of
     [BellMachover] p. 36, and Lemma 1.8 of [Mendelson] p. 36.  It is also "Our
     first proof" in Hirst and Hirst's _A Primer for Logic and Proof_ p. 17
     (PDF p. 23) at ~ http://www.appstate.edu/~~hirstjl/primer/hirst.pdf .
     Note that the second occurrence of ` ph ` in Steps 1 to 4 and the sixth in
     Step 3 may simultaneously be replaced by any wff ` ps ` , which may ease
     the understanding of the proof.  For a shorter version of the proof that
     takes advantage of previously proved theorems, see ~ id .  (Contributed by
     NM, 30-Sep-1992.)  (New usage is discouraged.)
     (Proof modification is discouraged.) $)
  id1 $p |- ( ph -> ph ) $=
    ( wi ax-1 ax-2 ax-mp ) AAABZBZFAACAFABBGFBAFCAFADEE $.

  $( Principle of identity with antecedent.  (Contributed by NM,
     26-Nov-1995.) $)
  idd $p |- ( ph -> ( ps -> ps ) ) $=
    ( wi id a1i ) BBCABDE $.

  ${
    a1d.1 $e |- ( ph -> ps ) $.
    $( Deduction introducing an embedded antecedent.  Deduction form of
       ~ ax-1 .  (Contributed by NM, 5-Jan-1993.)  (Proof shortened by Stefan
       Allan, 20-Mar-2006.) $)
    a1d $p |- ( ph -> ( ch -> ps ) ) $=
      ( wi ax-1 syl ) ABCBEDBCFG $.
  $}

  ${
    a2d.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( Deduction distributing an embedded antecedent.  Deduction form of
       ~ ax-2 .  (Contributed by NM, 23-Jun-1994.) $)
    a2d $p |- ( ph -> ( ( ps -> ch ) -> ( ps -> th ) ) ) $=
      ( wi ax-2 syl ) ABCDFFBCFBDFFEBCDGH $.
  $}

  ${
    a1ii.1 $e |- ch $.
    $( Add two antecedents to a wff.  (Contributed by Jeff Hankins,
       4-Aug-2009.)  (Proof modification is discouraged.) $)
    a1ii $p |- ( ph -> ( ps -> ch ) ) $=
      ( wi a1i ) BCEACBDFF $.
  $}

  ${
    a1iiOLD.1 $e |- ch $.
    $( Obsolete proof of ~ a1ii as of 21-Jul-2019.  Shorter proof, but uses one
       more axiom.  (Contributed by Wolf Lammen, 23-Jul-2013.)
       (New usage is discouraged.)  (Proof modification is discouraged.) $)
    a1iiOLD $p |- ( ph -> ( ps -> ch ) ) $=
      ( a1i a1d ) ACBCADEF $.
  $}

  ${
    sylcom.1 $e |- ( ph -> ( ps -> ch ) ) $.
    sylcom.2 $e |- ( ps -> ( ch -> th ) ) $.
    $( Syllogism inference with commutation of antecedents.  (Contributed by
       NM, 29-Aug-2004.)  (Proof shortened by Mel L. O'Cat, 2-Feb-2006.)
       (Proof shortened by Stefan Allan, 23-Feb-2006.) $)
    sylcom $p |- ( ph -> ( ps -> th ) ) $=
      ( wi a2i syl ) ABCGBDGEBCDFHI $.
  $}

  ${
    syl5com.1 $e |- ( ph -> ps ) $.
    syl5com.2 $e |- ( ch -> ( ps -> th ) ) $.
    $( Syllogism inference with commuted antecedents.  (Contributed by NM,
       24-May-2005.) $)
    syl5com $p |- ( ph -> ( ch -> th ) ) $=
      ( a1d sylcom ) ACBDABCEGFH $.
  $}

  ${
    $( Premise for ~ com12 .  See ~ pm2.04 for the theorem form. $)
    com12.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Inference that swaps (commutes) antecedents in an implication.
       Inference associated with ~ pm2.04 .  Its associated inference is
       ~ mpi .  (Contributed by NM, 29-Dec-1992.)  (Proof shortened by Wolf
       Lammen, 4-Aug-2012.) $)
    com12 $p |- ( ps -> ( ph -> ch ) ) $=
      ( id syl5com ) BBACBEDF $.
  $}

  ${
    syl5.1 $e |- ( ph -> ps ) $.
    syl5.2 $e |- ( ch -> ( ps -> th ) ) $.
    $( A syllogism rule of inference.  The first premise is used to replace the
       second antecedent of the second premise.  (Contributed by NM,
       27-Dec-1992.)  (Proof shortened by Wolf Lammen, 25-May-2013.) $)
    syl5 $p |- ( ch -> ( ph -> th ) ) $=
      ( syl5com com12 ) ACDABCDEFGH $.
  $}

  ${
    syl6.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl6.2 $e |- ( ch -> th ) $.
    $( A syllogism rule of inference.  The second premise is used to replace
       the consequent of the first premise.  (Contributed by NM, 5-Jan-1993.)
       (Proof shortened by Wolf Lammen, 30-Jul-2012.) $)
    syl6 $p |- ( ph -> ( ps -> th ) ) $=
      ( wi a1i sylcom ) ABCDECDGBFHI $.
  $}

  ${
    syl56.1 $e |- ( ph -> ps ) $.
    syl56.2 $e |- ( ch -> ( ps -> th ) ) $.
    syl56.3 $e |- ( th -> ta ) $.
    $( Combine ~ syl5 and ~ syl6 .  (Contributed by NM, 14-Nov-2013.) $)
    syl56 $p |- ( ch -> ( ph -> ta ) ) $=
      ( syl6 syl5 ) ABCEFCBDEGHIJ $.
  $}

  ${
    syl6com.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl6com.2 $e |- ( ch -> th ) $.
    $( Syllogism inference with commuted antecedents.  (Contributed by NM,
       25-May-2005.) $)
    syl6com $p |- ( ps -> ( ph -> th ) ) $=
      ( syl6 com12 ) ABDABCDEFGH $.
  $}

  ${
    mpcom.1 $e |- ( ps -> ph ) $.
    mpcom.2 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Modus ponens inference with commutation of antecedents.  (Contributed by
       NM, 17-Mar-1996.) $)
    mpcom $p |- ( ps -> ch ) $=
      ( com12 mpd ) BACDABCEFG $.
  $}

  ${
    syli.1 $e |- ( ps -> ( ph -> ch ) ) $.
    syli.2 $e |- ( ch -> ( ph -> th ) ) $.
    $( Syllogism inference with common nested antecedent.  (Contributed by NM,
       4-Nov-2004.) $)
    syli $p |- ( ps -> ( ph -> th ) ) $=
      ( com12 sylcom ) BACDECADFGH $.
  $}

  ${
    syl2im.1 $e |- ( ph -> ps ) $.
    syl2im.2 $e |- ( ch -> th ) $.
    syl2im.3 $e |- ( ps -> ( th -> ta ) ) $.
    $( Replace two antecedents.  Implication-only version of ~ syl2an .
       (Contributed by Wolf Lammen, 14-May-2013.) $)
    syl2im $p |- ( ph -> ( ch -> ta ) ) $=
      ( wi syl5 syl ) ABCEIFCDBEGHJK $.
  $}

  $( This theorem, called "Assertion," can be thought of as closed form of
     modus ponens ~ ax-mp .  Theorem *2.27 of [WhiteheadRussell] p. 104.
     (Contributed by NM, 15-Jul-1993.) $)
  pm2.27 $p |- ( ph -> ( ( ph -> ps ) -> ps ) ) $=
    ( wi id com12 ) ABCZABFDE $.

  ${
    mpdd.1 $e |- ( ph -> ( ps -> ch ) ) $.
    mpdd.2 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( A nested modus ponens deduction.  (Contributed by NM, 12-Dec-2004.) $)
    mpdd $p |- ( ph -> ( ps -> th ) ) $=
      ( wi a2d mpd ) ABCGBDGEABCDFHI $.
  $}

  ${
    mpid.1 $e |- ( ph -> ch ) $.
    mpid.2 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( A nested modus ponens deduction.  (Contributed by NM, 14-Dec-2004.) $)
    mpid $p |- ( ph -> ( ps -> th ) ) $=
      ( a1d mpdd ) ABCDACBEGFH $.
  $}

  ${
    mpdi.1 $e |- ( ps -> ch ) $.
    mpdi.2 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( A nested modus ponens deduction.  (Contributed by NM, 16-Apr-2005.)
       (Proof shortened by Mel L. O'Cat, 15-Jan-2008.) $)
    mpdi $p |- ( ph -> ( ps -> th ) ) $=
      ( wi a1i mpdd ) ABCDBCGAEHFI $.
  $}

  ${
    mpii.1 $e |- ch $.
    mpii.2 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( A doubly nested modus ponens inference.  (Contributed by NM,
       31-Dec-1993.)  (Proof shortened by Wolf Lammen, 31-Jul-2012.) $)
    mpii $p |- ( ph -> ( ps -> th ) ) $=
      ( a1i mpdi ) ABCDCBEGFH $.
  $}

  ${
    syld.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syld.2 $e |- ( ph -> ( ch -> th ) ) $.
    $( Syllogism deduction.  (Contributed by NM, 5-Aug-1993.)  (Proof shortened
       by Mel L. O'Cat, 19-Feb-2008.)  (Proof shortened by Wolf Lammen,
       3-Aug-2012.)

       Notice that ~ syld has the same form as ~ syl with ` ph ` added in front
       of each hypothesis and conclusion.  When all theorems referenced in a
       proof are converted in this way, we can replace ` ph ` with a hypothesis
       of the proof, allowing the hypothesis to be eliminated with ~ id and
       become an antecedent.  The Deduction Theorem for propositional calculus,
       e.g.  Theorem 3 in [Margaris] p. 56, tells us that this procedure is
       always possible. $)
    syld $p |- ( ph -> ( ps -> th ) ) $=
      ( wi a1d mpdd ) ABCDEACDGBFHI $.
  $}

  ${
    mp2d.1 $e |- ( ph -> ps ) $.
    mp2d.2 $e |- ( ph -> ch ) $.
    mp2d.3 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( A double modus ponens deduction.  (Contributed by NM, 23-May-2013.)
       (Proof shortened by Wolf Lammen, 23-Jul-2013.) $)
    mp2d $p |- ( ph -> th ) $=
      ( mpid mpd ) ABDEABCDFGHI $.
  $}

  ${
    a1dd.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction introducing a nested embedded antecedent.  (Contributed by NM,
       17-Dec-2004.)  (Proof shortened by Mel L. O'Cat, 15-Jan-2008.) $)
    a1dd $p |- ( ph -> ( ps -> ( th -> ch ) ) ) $=
      ( wi ax-1 syl6 ) ABCDCFECDGH $.
  $}

  ${
    pm2.43i.1 $e |- ( ph -> ( ph -> ps ) ) $.
    $( Inference absorbing redundant antecedent.  (Contributed by NM,
       10-Jan-1993.)  (Proof shortened by Mel L. O'Cat, 28-Nov-2008.) $)
    pm2.43i $p |- ( ph -> ps ) $=
      ( id mpd ) AABADCE $.
  $}

  ${
    pm2.43d.1 $e |- ( ph -> ( ps -> ( ps -> ch ) ) ) $.
    $( Deduction absorbing redundant antecedent.  (Contributed by NM,
       18-Aug-1993.)  (Proof shortened by Mel L. O'Cat, 28-Nov-2008.) $)
    pm2.43d $p |- ( ph -> ( ps -> ch ) ) $=
      ( id mpdi ) ABBCBEDF $.
  $}

  ${
    pm2.43a.1 $e |- ( ps -> ( ph -> ( ps -> ch ) ) ) $.
    $( Inference absorbing redundant antecedent.  (Contributed by NM,
       7-Nov-1995.)  (Proof shortened by Mel L. O'Cat, 28-Nov-2008.) $)
    pm2.43a $p |- ( ps -> ( ph -> ch ) ) $=
      ( id mpid ) BABCBEDF $.
  $}

  ${
    pm2.43b.1 $e |- ( ps -> ( ph -> ( ps -> ch ) ) ) $.
    $( Inference absorbing redundant antecedent.  (Contributed by NM,
       31-Oct-1995.) $)
    pm2.43b $p |- ( ph -> ( ps -> ch ) ) $=
      ( pm2.43a com12 ) BACABCDEF $.
  $}

  $( Absorption of redundant antecedent.  Also called the "Contraction" or
     "Hilbert" axiom.  Theorem *2.43 of [WhiteheadRussell] p. 106.
     (Contributed by NM, 10-Jan-1993.)  (Proof shortened by Mel L. O'Cat,
     15-Aug-2004.) $)
  pm2.43 $p |- ( ( ph -> ( ph -> ps ) ) -> ( ph -> ps ) ) $=
    ( wi pm2.27 a2i ) AABCBABDE $.

  ${
    imim2d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction adding nested antecedents.  (Contributed by NM,
       10-Jan-1993.) $)
    imim2d $p |- ( ph -> ( ( th -> ps ) -> ( th -> ch ) ) ) $=
      ( wi a1d a2d ) ADBCABCFDEGH $.
  $}

  $( A closed form of syllogism (see ~ syl ).  Theorem *2.05 of
     [WhiteheadRussell] p. 100.  (Contributed by NM, 29-Dec-1992.)  (Proof
     shortened by Wolf Lammen, 6-Sep-2012.) $)
  imim2 $p |- ( ( ph -> ps ) -> ( ( ch -> ph ) -> ( ch -> ps ) ) ) $=
    ( wi id imim2d ) ABDZABCGEF $.

  ${
    embantd.1 $e |- ( ph -> ps ) $.
    embantd.2 $e |- ( ph -> ( ch -> th ) ) $.
    $( Deduction embedding an antecedent.  (Contributed by Wolf Lammen,
       4-Oct-2013.) $)
    embantd $p |- ( ph -> ( ( ps -> ch ) -> th ) ) $=
      ( wi imim2d mpid ) ABCGBDEACDBFHI $.
  $}

  ${
    3syld.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3syld.2 $e |- ( ph -> ( ch -> th ) ) $.
    3syld.3 $e |- ( ph -> ( th -> ta ) ) $.
    $( Triple syllogism deduction.  (Contributed by Jeff Hankins,
       4-Aug-2009.) $)
    3syld $p |- ( ph -> ( ps -> ta ) ) $=
      ( syld ) ABDEABCDFGIHI $.
  $}

  ${
    sylsyld.1 $e |- ( ph -> ps ) $.
    sylsyld.2 $e |- ( ph -> ( ch -> th ) ) $.
    sylsyld.3 $e |- ( ps -> ( th -> ta ) ) $.
    $( A double syllogism inference.  (Contributed by Alan Sare,
       20-Apr-2011.) $)
    sylsyld $p |- ( ph -> ( ch -> ta ) ) $=
      ( wi syl syld ) ACDEGABDEIFHJK $.
  $}

  ${
    imim12i.1 $e |- ( ph -> ps ) $.
    imim12i.2 $e |- ( ch -> th ) $.
    $( Inference joining two implications.  (Contributed by NM, 12-Mar-1993.)
       (Proof shortened by Mel L. O'Cat, 29-Oct-2011.) $)
    imim12i $p |- ( ( ps -> ch ) -> ( ph -> th ) ) $=
      ( wi imim2i syl5 ) ABBCGDECDBFHI $.
  $}

  ${
    imim1i.1 $e |- ( ph -> ps ) $.
    $( Inference adding common consequents in an implication, thereby
       interchanging the original antecedent and consequent.  (Contributed by
       NM, 28-Dec-1992.)  (Proof shortened by Wolf Lammen, 4-Aug-2012.) $)
    imim1i $p |- ( ( ps -> ch ) -> ( ph -> ch ) ) $=
      ( id imim12i ) ABCCDCEF $.
  $}

  ${
    imim3i.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Inference adding three nested antecedents.  (Contributed by NM,
       19-Dec-2006.) $)
    imim3i $p |- ( ( th -> ph ) -> ( ( th -> ps ) -> ( th -> ch ) ) ) $=
      ( wi imim2i a2d ) DAFDBCABCFDEGH $.
  $}

  ${
    sylc.1 $e |- ( ph -> ps ) $.
    sylc.2 $e |- ( ph -> ch ) $.
    sylc.3 $e |- ( ps -> ( ch -> th ) ) $.
    $( A syllogism inference combined with contraction.  (Contributed by NM,
       4-May-1994.)  (Revised by NM, 13-Jul-2013.) $)
    sylc $p |- ( ph -> th ) $=
      ( syl2im pm2.43i ) ADABACDEFGHI $.
  $}

  ${
    syl3c.1 $e |- ( ph -> ps ) $.
    syl3c.2 $e |- ( ph -> ch ) $.
    syl3c.3 $e |- ( ph -> th ) $.
    syl3c.4 $e |- ( ps -> ( ch -> ( th -> ta ) ) ) $.
    $( A syllogism inference combined with contraction.  (Contributed by Alan
       Sare, 7-Jul-2011.) $)
    syl3c $p |- ( ph -> ta ) $=
      ( wi sylc mpd ) ADEHABCDEJFGIKL $.
  $}

  ${
    syl6mpi.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl6mpi.2 $e |- th $.
    syl6mpi.3 $e |- ( ch -> ( th -> ta ) ) $.
    $( A syllogism inference.  (Contributed by Alan Sare, 8-Jul-2011.)  (Proof
       shortened by Wolf Lammen, 13-Sep-2012.) $)
    syl6mpi $p |- ( ph -> ( ps -> ta ) ) $=
      ( mpi syl6 ) ABCEFCDEGHIJ $.
  $}

  ${
    mpsyl.1 $e |- ph $.
    mpsyl.2 $e |- ( ps -> ch ) $.
    mpsyl.3 $e |- ( ph -> ( ch -> th ) ) $.
    $( Modus ponens combined with a syllogism inference.  (Contributed by Alan
       Sare, 20-Apr-2011.) $)
    mpsyl $p |- ( ps -> th ) $=
      ( a1i sylc ) BACDABEHFGI $.
  $}

  ${
    syl6c.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl6c.2 $e |- ( ph -> ( ps -> th ) ) $.
    syl6c.3 $e |- ( ch -> ( th -> ta ) ) $.
    $( Inference combining ~ syl6 with contraction.  (Contributed by Alan Sare,
       2-May-2011.) $)
    syl6c $p |- ( ph -> ( ps -> ta ) ) $=
      ( wi syl6 mpdd ) ABDEGABCDEIFHJK $.
  $}

  ${
    syl6ci.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl6ci.2 $e |- ( ph -> th ) $.
    syl6ci.3 $e |- ( ch -> ( th -> ta ) ) $.
    $( A syllogism inference combined with contraction.  (Contributed by Alan
       Sare, 18-Mar-2012.) $)
    syl6ci $p |- ( ph -> ( ps -> ta ) ) $=
      ( a1d syl6c ) ABCDEFADBGIHJ $.
  $}

  ${
    syldd.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    syldd.2 $e |- ( ph -> ( ps -> ( th -> ta ) ) ) $.
    $( Nested syllogism deduction.  (Contributed by NM, 12-Dec-2004.)  (Proof
       shortened by Wolf Lammen, 11-May-2013.) $)
    syldd $p |- ( ph -> ( ps -> ( ch -> ta ) ) ) $=
      ( wi imim2 syl6c ) ABDEHCDHCEHGFDECIJ $.
  $}

  ${
    syl5d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl5d.2 $e |- ( ph -> ( th -> ( ch -> ta ) ) ) $.
    $( A nested syllogism deduction.  (Contributed by NM, 14-May-1993.)  (Proof
       shortened by Josh Purinton, 29-Dec-2000.)  (Proof shortened by Mel L.
       O'Cat, 2-Feb-2006.) $)
    syl5d $p |- ( ph -> ( th -> ( ps -> ta ) ) ) $=
      ( wi a1d syldd ) ADBCEABCHDFIGJ $.
  $}

  ${
    syl7.1 $e |- ( ph -> ps ) $.
    syl7.2 $e |- ( ch -> ( th -> ( ps -> ta ) ) ) $.
    $( A syllogism rule of inference.  The first premise is used to replace the
       third antecedent of the second premise.  (Contributed by NM,
       12-Jan-1993.)  (Proof shortened by Wolf Lammen, 3-Aug-2012.) $)
    syl7 $p |- ( ch -> ( th -> ( ph -> ta ) ) ) $=
      ( wi a1i syl5d ) CABDEABHCFIGJ $.
  $}

  ${
    syl6d.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    syl6d.2 $e |- ( ph -> ( th -> ta ) ) $.
    $( A nested syllogism deduction.  (Contributed by NM, 11-May-1993.)  (Proof
       shortened by Josh Purinton, 29-Dec-2000.)  (Proof shortened by Mel L.
       O'Cat, 2-Feb-2006.) $)
    syl6d $p |- ( ph -> ( ps -> ( ch -> ta ) ) ) $=
      ( wi a1d syldd ) ABCDEFADEHBGIJ $.
  $}

  ${
    syl8.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    syl8.2 $e |- ( th -> ta ) $.
    $( A syllogism rule of inference.  The second premise is used to replace
       the consequent of the first premise.  (Contributed by NM, 1-Aug-1994.)
       (Proof shortened by Wolf Lammen, 3-Aug-2012.) $)
    syl8 $p |- ( ph -> ( ps -> ( ch -> ta ) ) ) $=
      ( wi a1i syl6d ) ABCDEFDEHAGIJ $.
  $}

  ${
    syl9.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl9.2 $e |- ( th -> ( ch -> ta ) ) $.
    $( A nested syllogism inference with different antecedents.  (Contributed
       by NM, 13-May-1993.)  (Proof shortened by Josh Purinton,
       29-Dec-2000.) $)
    syl9 $p |- ( ph -> ( th -> ( ps -> ta ) ) ) $=
      ( wi a1i syl5d ) ABCDEFDCEHHAGIJ $.
  $}

  ${
    syl9r.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl9r.2 $e |- ( th -> ( ch -> ta ) ) $.
    $( A nested syllogism inference with different antecedents.  (Contributed
       by NM, 14-May-1993.) $)
    syl9r $p |- ( th -> ( ph -> ( ps -> ta ) ) ) $=
      ( wi syl9 com12 ) ADBEHABCDEFGIJ $.
  $}

  ${
    syl10.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl10.2 $e |- ( ph -> ( ps -> ( th -> ta ) ) ) $.
    syl10.3 $e |- ( ch -> ( ta -> et ) ) $.
    $( A nested syllogism inference.  (Contributed by Alan Sare,
       17-Jul-2011.) $)
    syl10 $p |- ( ph -> ( ps -> ( th -> et ) ) ) $=
      ( wi syl6 syldd ) ABDEFHABCEFJGIKL $.
  $}

  ${
    imim12d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    imim12d.2 $e |- ( ph -> ( th -> ta ) ) $.
    $( Deduction combining antecedents and consequents.  (Contributed by NM,
       7-Aug-1994.)  (Proof shortened by Mel L. O'Cat, 30-Oct-2011.) $)
    imim12d $p |- ( ph -> ( ( ch -> th ) -> ( ps -> ta ) ) ) $=
      ( wi imim2d syl5d ) ABCCDHEFADECGIJ $.
  $}

  ${
    imim1d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction adding nested consequents.  (Contributed by NM, 3-Apr-1994.)
       (Proof shortened by Wolf Lammen, 12-Sep-2012.) $)
    imim1d $p |- ( ph -> ( ( ch -> th ) -> ( ps -> th ) ) ) $=
      ( idd imim12d ) ABCDDEADFG $.
  $}

  $( A closed form of syllogism (see ~ syl ).  Theorem *2.06 of
     [WhiteheadRussell] p. 100.  (Contributed by NM, 29-Dec-1992.)  (Proof
     shortened by Wolf Lammen, 25-May-2013.) $)
  imim1 $p |- ( ( ph -> ps ) -> ( ( ps -> ch ) -> ( ph -> ch ) ) ) $=
    ( wi id imim1d ) ABDZABCGEF $.

  $( Theorem *2.83 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.83 $p |- ( ( ph -> ( ps -> ch ) )
      -> ( ( ph -> ( ch -> th ) ) -> ( ph -> ( ps -> th ) ) ) ) $=
    ( wi imim1 imim3i ) BCECDEBDEABCDFG $.

  ${
    com3.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( Commutation of antecedents.  Swap 2nd and 3rd.  (Contributed by NM,
       27-Dec-1992.)  (Proof shortened by Wolf Lammen, 4-Aug-2012.) $)
    com23 $p |- ( ph -> ( ch -> ( ps -> th ) ) ) $=
      ( wi pm2.27 syl9 ) ABCDFCDECDGH $.

    $( Commutation of antecedents.  Rotate right.  (Contributed by NM,
       25-Apr-1994.) $)
    com3r $p |- ( ch -> ( ph -> ( ps -> th ) ) ) $=
      ( wi com23 com12 ) ACBDFABCDEGH $.

    $( Commutation of antecedents.  Swap 1st and 3rd.  (Contributed by NM,
       25-Apr-1994.)  (Proof shortened by Wolf Lammen, 28-Jul-2012.) $)
    com13 $p |- ( ch -> ( ps -> ( ph -> th ) ) ) $=
      ( com3r com23 ) CABDABCDEFG $.

    $( Commutation of antecedents.  Rotate left.  (Contributed by NM,
       25-Apr-1994.)  (Proof shortened by Wolf Lammen, 28-Jul-2012.) $)
    com3l $p |- ( ps -> ( ch -> ( ph -> th ) ) ) $=
      ( com3r ) CABDABCDEFF $.
  $}

  $( Swap antecedents.  Theorem *2.04 of [WhiteheadRussell] p. 100.  This was
     the third axiom in Frege's logic system, specifically Proposition 8 of
     [Frege1879] p. 35.  Its associated inference is ~ com12 .  (Contributed by
     NM, 27-Dec-1992.)  (Proof shortened by Wolf Lammen, 12-Sep-2012.) $)
  pm2.04 $p |- ( ( ph -> ( ps -> ch ) ) -> ( ps -> ( ph -> ch ) ) ) $=
    ( wi id com23 ) ABCDDZABCGEF $.

  ${
    com4.1 $e |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $.
    $( Commutation of antecedents.  Swap 3rd and 4th.  (Contributed by NM,
       25-Apr-1994.) $)
    com34 $p |- ( ph -> ( ps -> ( th -> ( ch -> ta ) ) ) ) $=
      ( wi pm2.04 syl6 ) ABCDEGGDCEGGFCDEHI $.

    $( Commutation of antecedents.  Rotate left.  (Contributed by NM,
       25-Apr-1994.)  (Proof shortened by Mel L. O'Cat, 15-Aug-2004.) $)
    com4l $p |- ( ps -> ( ch -> ( th -> ( ph -> ta ) ) ) ) $=
      ( wi com3l com34 ) BCADEABCDEGFHI $.

    $( Commutation of antecedents.  Rotate twice.  (Contributed by NM,
       25-Apr-1994.) $)
    com4t $p |- ( ch -> ( th -> ( ph -> ( ps -> ta ) ) ) ) $=
      ( com4l ) BCDAEABCDEFGG $.

    $( Commutation of antecedents.  Rotate right.  (Contributed by NM,
       25-Apr-1994.) $)
    com4r $p |- ( th -> ( ph -> ( ps -> ( ch -> ta ) ) ) ) $=
      ( com4t com4l ) CDABEABCDEFGH $.

    $( Commutation of antecedents.  Swap 2nd and 4th.  (Contributed by NM,
       25-Apr-1994.)  (Proof shortened by Wolf Lammen, 28-Jul-2012.) $)
    com24 $p |- ( ph -> ( th -> ( ch -> ( ps -> ta ) ) ) ) $=
      ( wi com4t com13 ) CDABEGABCDEFHI $.

    $( Commutation of antecedents.  Swap 1st and 4th.  (Contributed by NM,
       25-Apr-1994.)  (Proof shortened by Wolf Lammen, 28-Jul-2012.) $)
    com14 $p |- ( th -> ( ps -> ( ch -> ( ph -> ta ) ) ) ) $=
      ( wi com4l com3r ) BCDAEGABCDEFHI $.
  $}

  ${
    com5.1 $e |- ( ph -> ( ps -> ( ch -> ( th -> ( ta -> et ) ) ) ) ) $.
    $( Commutation of antecedents.  Swap 4th and 5th.  (Contributed by Jeff
       Hankins, 28-Jun-2009.) $)
    com45 $p |- ( ph -> ( ps -> ( ch -> ( ta -> ( th -> et ) ) ) ) ) $=
      ( wi pm2.04 syl8 ) ABCDEFHHEDFHHGDEFIJ $.

    $( Commutation of antecedents.  Swap 3rd and 5th.  (Contributed by Jeff
       Hankins, 28-Jun-2009.) $)
    com35 $p |- ( ph -> ( ps -> ( ta -> ( th -> ( ch -> et ) ) ) ) ) $=
      ( wi com34 com45 ) ABDECFHABDCEFABCDEFHGIJI $.

    $( Commutation of antecedents.  Swap 2nd and 5th.  (Contributed by Jeff
       Hankins, 28-Jun-2009.) $)
    com25 $p |- ( ph -> ( ta -> ( ch -> ( th -> ( ps -> et ) ) ) ) ) $=
      ( wi com24 com45 ) ADCEBFHADCBEFABCDEFHGIJI $.

    $( Commutation of antecedents.  Rotate left.  (Contributed by Jeff Hankins,
       28-Jun-2009.)  (Proof shortened by Wolf Lammen, 29-Jul-2012.) $)
    com5l $p |- ( ps -> ( ch -> ( th -> ( ta -> ( ph -> et ) ) ) ) ) $=
      ( wi com4l com45 ) BCDAEFABCDEFHGIJ $.

    $( Commutation of antecedents.  Swap 1st and 5th.  (Contributed by Jeff
       Hankins, 28-Jun-2009.)  (Proof shortened by Wolf Lammen,
       29-Jul-2012.) $)
    com15 $p |- ( ta -> ( ps -> ( ch -> ( th -> ( ph -> et ) ) ) ) ) $=
      ( wi com5l com4r ) BCDEAFHABCDEFGIJ $.

    $( Commutation of antecedents.  Rotate left twice.  (Contributed by Jeff
       Hankins, 28-Jun-2009.) $)
    com52l $p |- ( ch -> ( th -> ( ta -> ( ph -> ( ps -> et ) ) ) ) ) $=
      ( com5l ) BCDEAFABCDEFGHH $.

    $( Commutation of antecedents.  Rotate right twice.  (Contributed by Jeff
       Hankins, 28-Jun-2009.) $)
    com52r $p |- ( th -> ( ta -> ( ph -> ( ps -> ( ch -> et ) ) ) ) ) $=
      ( com52l com5l ) CDEABFABCDEFGHI $.

    $( Commutation of antecedents.  Rotate right.  (Contributed by Wolf Lammen,
       29-Jul-2012.) $)
    com5r $p |- ( ta -> ( ph -> ( ps -> ( ch -> ( th -> et ) ) ) ) ) $=
      ( com52l ) CDEABFABCDEFGHH $.
  $}

  $( Closed form of ~ imim12i and of ~ 3syl .  (Contributed by BJ,
     16-Jul-2019.) $)
  imim12 $p |- ( ( ph -> ps ) ->
                      ( ( ch -> th ) -> ( ( ps -> ch ) -> ( ph -> th ) ) ) ) $=
    ( wi imim2 com13 imim2i com24 ) ABEABCEZCDEZDBJKDEEAKJBDCDBFGHI $.

  $( Elimination of a nested antecedent as a kind of reversal of inference
     ~ ja .  (Contributed by Wolf Lammen, 9-May-2013.) $)
  jarr $p |- ( ( ( ph -> ps ) -> ch ) -> ( ps -> ch ) ) $=
    ( wi ax-1 imim1i ) BABDCBAEF $.

  ${
    pm2.86d.1 $e |- ( ph -> ( ( ps -> ch ) -> ( ps -> th ) ) ) $.
    $( Deduction based on ~ pm2.86 .  (Contributed by NM, 29-Jun-1995.)  (Proof
       shortened by Wolf Lammen, 3-Apr-2013.) $)
    pm2.86d $p |- ( ph -> ( ps -> ( ch -> th ) ) ) $=
      ( wi ax-1 syl5 com23 ) ACBDCBCFABDFCBGEHI $.
  $}

  $( Converse of axiom ~ ax-2 .  Theorem *2.86 of [WhiteheadRussell] p. 108.
     (Contributed by NM, 25-Apr-1994.)  (Proof shortened by Wolf Lammen,
     3-Apr-2013.) $)
  pm2.86 $p |- ( ( ( ph -> ps ) -> ( ph -> ch ) ) ->
                                                    ( ph -> ( ps -> ch ) ) ) $=
    ( wi id pm2.86d ) ABDACDDZABCGEF $.

  ${
    pm2.86i.1 $e |- ( ( ph -> ps ) -> ( ph -> ch ) ) $.
    $( Inference associated with ~ pm2.86 .  (Contributed by NM, 5-Aug-1993.)
       (Proof shortened by Wolf Lammen, 3-Apr-2013.) $)
    pm2.86i $p |- ( ph -> ( ps -> ch ) ) $=
      ( wi ax-1 syl com12 ) BACBABEACEBAFDGH $.

    $( Alternate proof of ~ pm2.86i with only three essential steps.
       (Contributed by NM, 5-Aug-1993.)  (Revised by BJ, 19-Jul-2019.)
       (Proof modification is discouraged.)  (New usage is discouraged.) $)
    pm2.86iALT $p |- ( ph -> ( ps -> ch ) ) $=
      ( wi pm2.86 ax-mp ) ABEACEEABCEEDABCFG $.
  $}

  $( The Linearity Axiom of the infinite-valued sentential logic (L-infinity)
     of Lukasiewicz.  See ~ loowoz for an alternate axiom.  (Contributed by Mel
     L. O'Cat, 12-Aug-2004.)  (Proof modification is discouraged.) $)
  loolin $p |- ( ( ( ph -> ps ) -> ( ps -> ph ) ) -> ( ps -> ph ) ) $=
    ( wi jarr pm2.43d ) ABCBACZCBAABFDE $.

  $( An alternate for the Linearity Axiom of the infinite-valued sentential
     logic (L-infinity) of Lukasiewicz ~ loolin , due to Barbara Wozniakowska,
     _Reports on Mathematical Logic_ 10, 129-137 (1978).  (Contributed by Mel
     L. O'Cat, 8-Aug-2004.) $)
  loowoz $p |- ( ( ( ph -> ps ) -> ( ph -> ch ) )
      -> ( ( ps -> ph ) -> ( ps -> ch ) ) ) $=
    ( wi jarr a2d ) ABDACDZDBACABGEF $.

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Logical negation
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  This section makes our first use of the third axiom of propositional
  calculus, ~ ax-3 .

$)

  ${
    con4d.1 $e |- ( ph -> ( -. ps -> -. ch ) ) $.
    $( Deduction derived from axiom ~ ax-3 .  (Contributed by NM,
       26-Mar-1995.) $)
    con4d $p |- ( ph -> ( ch -> ps ) ) $=
      ( wn wi ax-3 syl ) ABECEFCBFDBCGH $.
  $}

  ${
    pm2.21d.1 $e |- ( ph -> -. ps ) $.
    $( A contradiction implies anything.  Deduction from ~ pm2.21 .
       (Contributed by NM, 10-Feb-1996.) $)
    pm2.21d $p |- ( ph -> ( ps -> ch ) ) $=
      ( wn a1d con4d ) ACBABECEDFG $.
  $}

  ${
    pm2.21ddOLD.1 $e |- ( ph -> ps ) $.
    pm2.21ddOLD.2 $e |- ( ph -> -. ps ) $.
    $( Obsolete proof of ~ pm2.21dd as of 22-Jul-2019.  (Contributed by Mario
       Carneiro, 9-Feb-2017.)  (Proof modification is discouraged.)
       (New usage is discouraged.) $)
    pm2.21ddOLD $p |- ( ph -> ch ) $=
      ( pm2.21d mpd ) ABCDABCEFG $.
  $}

  $( From a wff and its negation, anything is true.  Theorem *2.21 of
     [WhiteheadRussell] p. 104.  Also called the Duns Scotus law.  (Contributed
     by NM, 29-Dec-1992.)  (Proof shortened by Wolf Lammen, 14-Sep-2012.) $)
  pm2.21 $p |- ( -. ph -> ( ph -> ps ) ) $=
    ( wn id pm2.21d ) ACZABFDE $.

  $( Theorem *2.24 of [WhiteheadRussell] p. 104.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.24 $p |- ( ph -> ( -. ph -> ps ) ) $=
    ( wn pm2.21 com12 ) ACABABDE $.

  $( Proof by contradiction.  Theorem *2.18 of [WhiteheadRussell] p. 103.  Also
     called the Law of Clavius.  See also ~ pm2.01 .  (Contributed by NM,
     29-Dec-1992.) $)
  pm2.18 $p |- ( ( -. ph -> ph ) -> ph ) $=
    ( wn wi pm2.21 a2i con4d pm2.43i ) ABZACZAIAIHAIBZAJDEFG $.

  ${
    pm2.18d.1 $e |- ( ph -> ( -. ps -> ps ) ) $.
    $( Deduction based on reductio ad absurdum.  (Contributed by FL,
       12-Jul-2009.)  (Proof shortened by Andrew Salmon, 7-May-2011.) $)
    pm2.18d $p |- ( ph -> ps ) $=
      ( wn wi pm2.18 syl ) ABDBEBCBFG $.
  $}

  $( Converse of double negation.  Theorem *2.14 of [WhiteheadRussell] p. 102.
     This was the fifth axiom of Frege, specifically Proposition 31 of
     [Frege1879] p. 44.  In classical logic (our logic) this is always true.
     In intuitionistic logic this is not always true; in intuitionistic logic,
     when this is true for some ` ph ` , then ` ph ` is stable.  (Contributed
     by NM, 29-Dec-1992.)  (Proof shortened by David Harvey, 5-Sep-1999.)
     (Proof shortened by Josh Purinton, 29-Dec-2000.) $)
  notnot2 $p |- ( -. -. ph -> ph ) $=
    ( wn pm2.21 pm2.18d ) ABZBAEACD $.

  ${
    notnotrd.1 $e |- ( ph -> -. -. ps ) $.
    $( Deduction converting double-negation into the original wff, aka the
       double negation rule.  A translation of natural deduction rule ` -. -. `
       -C, ` _G |- -. -. ps ` => ` _G |- ps ` ; see ~ natded .  This is
       definition NNC in [Pfenning] p. 17.  This rule is valid in classical
       logic (which MPE uses), but not intuitionistic logic.  (Contributed by
       DAW, 8-Feb-2017.) $)
    notnotrd $p |- ( ph -> ps ) $=
      ( wn notnot2 syl ) ABDDBCBEF $.
  $}

  ${
    notnotri.1 $e |- -. -. ph $.
    $( Inference from double negation.  (Contributed by NM, 27-Feb-2008.) $)
    notnotri $p |- ph $=
      ( wn notnot2 ax-mp ) ACCABADE $.
  $}

  ${
    con2d.1 $e |- ( ph -> ( ps -> -. ch ) ) $.
    $( A contraposition deduction.  (Contributed by NM, 19-Aug-1993.) $)
    con2d $p |- ( ph -> ( ch -> -. ps ) ) $=
      ( wn notnot2 syl5 con4d ) ABEZCIEBACEBFDGH $.
  $}

  $( Contraposition.  Theorem *2.03 of [WhiteheadRussell] p. 100.  (Contributed
     by NM, 29-Dec-1992.)  (Proof shortened by Wolf Lammen, 12-Feb-2013.) $)
  con2 $p |- ( ( ph -> -. ps ) -> ( ps -> -. ph ) ) $=
    ( wn wi id con2d ) ABCDZABGEF $.

  ${
    mt2d.1 $e |- ( ph -> ch ) $.
    mt2d.2 $e |- ( ph -> ( ps -> -. ch ) ) $.
    $( Modus tollens deduction.  (Contributed by NM, 4-Jul-1994.) $)
    mt2d $p |- ( ph -> -. ps ) $=
      ( wn con2d mpd ) ACBFDABCEGH $.
  $}

  ${
    mt2i.1 $e |- ch $.
    mt2i.2 $e |- ( ph -> ( ps -> -. ch ) ) $.
    $( Modus tollens inference.  (Contributed by NM, 26-Mar-1995.)  (Proof
       shortened by Wolf Lammen, 15-Sep-2012.) $)
    mt2i $p |- ( ph -> -. ps ) $=
      ( a1i mt2d ) ABCCADFEG $.
  $}

  ${
    nsyl3.1 $e |- ( ph -> -. ps ) $.
    nsyl3.2 $e |- ( ch -> ps ) $.
    $( A negated syllogism inference.  (Contributed by NM, 1-Dec-1995.) $)
    nsyl3 $p |- ( ch -> -. ph ) $=
      ( wn wi a1i mt2d ) CABEABFGCDHI $.
  $}

  ${
    con2i.a $e |- ( ph -> -. ps ) $.
    $( A contraposition inference.  Its associated inference is ~ mt2 .
       (Contributed by NM, 10-Jan-1993.)  (Proof shortened by Mel L. O'Cat,
       28-Nov-2008.)  (Proof shortened by Wolf Lammen, 13-Jun-2013.) $)
    con2i $p |- ( ps -> -. ph ) $=
      ( id nsyl3 ) ABBCBDE $.
  $}

  ${
    nsyl.1 $e |- ( ph -> -. ps ) $.
    nsyl.2 $e |- ( ch -> ps ) $.
    $( A negated syllogism inference.  (Contributed by NM, 31-Dec-1993.)
       (Proof shortened by Wolf Lammen, 2-Mar-2013.) $)
    nsyl $p |- ( ph -> -. ch ) $=
      ( nsyl3 con2i ) CAABCDEFG $.
  $}

  $( Converse of double negation.  Theorem *2.12 of [WhiteheadRussell] p. 101.
     This was the sixth axiom of Frege, specifically Proposition 41 of
     [Frege1879] p. 47.  (Contributed by NM, 28-Dec-1992.)  (Proof shortened by
     Wolf Lammen, 2-Mar-2013.) $)
  notnot1 $p |- ( ph -> -. -. ph ) $=
    ( wn id con2i ) ABZAECD $.

  ${
    negbi.1 $e |- ph $.
    $( Infer double negation.  (Contributed by NM, 27-Feb-2008.) $)
    notnoti $p |- -. -. ph $=
      ( wn notnot1 ax-mp ) AACCBADE $.
  $}

  ${
    con1d.1 $e |- ( ph -> ( -. ps -> ch ) ) $.
    $( A contraposition deduction.  (Contributed by NM, 27-Dec-1992.) $)
    con1d $p |- ( ph -> ( -. ch -> ps ) ) $=
      ( wn notnot1 syl6 con4d ) ABCEZABECIEDCFGH $.
  $}

  ${
    mt3d.1 $e |- ( ph -> -. ch ) $.
    mt3d.2 $e |- ( ph -> ( -. ps -> ch ) ) $.
    $( Modus tollens deduction.  (Contributed by NM, 26-Mar-1995.) $)
    mt3d $p |- ( ph -> ps ) $=
      ( wn con1d mpd ) ACFBDABCEGH $.
  $}

  ${
    mt3i.1 $e |- -. ch $.
    mt3i.2 $e |- ( ph -> ( -. ps -> ch ) ) $.
    $( Modus tollens inference.  (Contributed by NM, 26-Mar-1995.)  (Proof
       shortened by Wolf Lammen, 15-Sep-2012.) $)
    mt3i $p |- ( ph -> ps ) $=
      ( wn a1i mt3d ) ABCCFADGEH $.
  $}

  ${
    nsyl2.1 $e |- ( ph -> -. ps ) $.
    nsyl2.2 $e |- ( -. ch -> ps ) $.
    $( A negated syllogism inference.  (Contributed by NM, 26-Jun-1994.) $)
    nsyl2 $p |- ( ph -> ch ) $=
      ( wn wi a1i mt3d ) ACBDCFBGAEHI $.
  $}

  $( Contraposition.  Theorem *2.15 of [WhiteheadRussell] p. 102.  (Contributed
     by NM, 29-Dec-1992.)  (Proof shortened by Wolf Lammen, 12-Feb-2013.) $)
  con1 $p |- ( ( -. ph -> ps ) -> ( -. ps -> ph ) ) $=
    ( wn wi id con1d ) ACBDZABGEF $.

  ${
    con1i.a $e |- ( -. ph -> ps ) $.
    $( A contraposition inference.  Its associated inference is ~ mt3 .
       (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Mel L. O'Cat,
       28-Nov-2008.)  (Proof shortened by Wolf Lammen, 19-Jun-2013.) $)
    con1i $p |- ( -. ps -> ph ) $=
      ( wn id nsyl2 ) BDZBAGECF $.
  $}

  ${
    con4i.1 $e |- ( -. ph -> -. ps ) $.
    $( Inference rule derived from axiom ~ ax-3 .  Its associated inference is
       ~ mt4 .  (Contributed by NM, 29-Dec-1992.)  (Proof shortened by Wolf
       Lammen, 21-Jun-2013.) $)
    con4i $p |- ( ps -> ph ) $=
      ( wn notnot1 nsyl2 ) BBDABECF $.
  $}

  ${
    pm2.21i.1 $e |- -. ph $.
    $( A contradiction implies anything.  Inference associated with ~ pm2.21 .
       (Contributed by NM, 16-Sep-1993.) $)
    pm2.21i $p |- ( ph -> ps ) $=
      ( wn a1i con4i ) BAADBDCEF $.
  $}

  ${
    pm2.24ii.1 $e |- ph $.
    pm2.24ii.2 $e |- -. ph $.
    $( A contradiction implies anything.  Inference associated with ~ pm2.21i
       and ~ pm2.24i .  (Contributed by NM, 27-Feb-2008.) $)
    pm2.24ii $p |- ps $=
      ( pm2.21i ax-mp ) ABCABDEF $.
  $}

  ${
    con3d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( A contraposition deduction.  Deduction form of ~ con3 .  (Contributed by
       NM, 10-Jan-1993.) $)
    con3d $p |- ( ph -> ( -. ch -> -. ps ) ) $=
      ( wn notnot2 syl5 con1d ) ABEZCIEBACBFDGH $.
  $}

  $( Contraposition.  Theorem *2.16 of [WhiteheadRussell] p. 103.  This was the
     fourth axiom of Frege, specifically Proposition 28 of [Frege1879] p. 43.
     (Contributed by NM, 29-Dec-1992.)  (Proof shortened by Wolf Lammen,
     13-Feb-2013.) $)
  con3 $p |- ( ( ph -> ps ) -> ( -. ps -> -. ph ) ) $=
    ( wi id con3d ) ABCZABFDE $.

  ${
    con3i.a $e |- ( ph -> ps ) $.
    $( A contraposition inference.  Inference associated with ~ con3 .  Its
       associated inference is ~ mto .  (Contributed by NM, 3-Jan-1993.)
       (Proof shortened by Wolf Lammen, 20-Jun-2013.) $)
    con3i $p |- ( -. ps -> -. ph ) $=
      ( wn id nsyl ) BDZBAGECF $.
  $}

  ${
    con3rr3.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Rotate through consequent right.  (Contributed by Wolf Lammen,
       3-Nov-2013.) $)
    con3rr3 $p |- ( -. ch -> ( ph -> -. ps ) ) $=
      ( wn con3d com12 ) ACEBEABCDFG $.
  $}

  ${
    mt4.1 $e |- ph $.
    mt4.2 $e |- ( -. ps -> -. ph ) $.
    $( The rule of modus tollens.  Inference associated with ~ con4i .
       (Contributed by Wolf Lammen, 12-May-2013.) $)
    mt4 $p |- ps $=
      ( con4i ax-mp ) ABCBADEF $.
  $}

  ${
    mt4d.1 $e |- ( ph -> ps ) $.
    mt4d.2 $e |- ( ph -> ( -. ch -> -. ps ) ) $.
    $( Modus tollens deduction.  Deduction form of ~ mt4 .  (Contributed by NM,
       9-Jun-2006.) $)
    mt4d $p |- ( ph -> ch ) $=
      ( con4d mpd ) ABCDACBEFG $.
  $}

  ${
    mt4i.1 $e |- ch $.
    mt4i.2 $e |- ( ph -> ( -. ps -> -. ch ) ) $.
    $( Modus tollens inference.  (Contributed by Wolf Lammen, 12-May-2013.) $)
    mt4i $p |- ( ph -> ps ) $=
      ( a1i mt4d ) ACBCADFEG $.
  $}

  ${
    nsyld.1 $e |- ( ph -> ( ps -> -. ch ) ) $.
    nsyld.2 $e |- ( ph -> ( ta -> ch ) ) $.
    $( A negated syllogism deduction.  (Contributed by NM, 9-Apr-2005.) $)
    nsyld $p |- ( ph -> ( ps -> -. ta ) ) $=
      ( wn con3d syld ) ABCGDGEADCFHI $.
  $}

  ${
    nsyli.1 $e |- ( ph -> ( ps -> ch ) ) $.
    nsyli.2 $e |- ( th -> -. ch ) $.
    $( A negated syllogism inference.  (Contributed by NM, 3-May-1994.) $)
    nsyli $p |- ( ph -> ( th -> -. ps ) ) $=
      ( wn con3d syl5 ) DCGABGFABCEHI $.
  $}

  ${
    nsyl4.1 $e |- ( ph -> ps ) $.
    nsyl4.2 $e |- ( -. ph -> ch ) $.
    $( A negated syllogism inference.  (Contributed by NM, 15-Feb-1996.) $)
    nsyl4 $p |- ( -. ch -> ps ) $=
      ( wn con1i syl ) CFABACEGDH $.
  $}

  ${
    pm2.24d.1 $e |- ( ph -> ps ) $.
    $( Deduction version of ~ pm2.24 .  (Contributed by NM, 30-Jan-2006.) $)
    pm2.24d $p |- ( ph -> ( -. ps -> ch ) ) $=
      ( wn a1d con1d ) ACBABCEDFG $.
  $}

  ${
    pm2.24i.1 $e |- ph $.
    $( Inference version of ~ pm2.24 .  (Contributed by NM, 20-Aug-2001.) $)
    pm2.24i $p |- ( -. ph -> ps ) $=
      ( wn a1i con1i ) BAABDCEF $.
  $}

  $( Theorem *3.2 of [WhiteheadRussell] p. 111, expressed with primitive
     connectives.  (Contributed by NM, 29-Dec-1992.)  (Proof shortened by Josh
     Purinton, 29-Dec-2000.) $)
  pm3.2im $p |- ( ph -> ( ps -> -. ( ph -> -. ps ) ) ) $=
    ( wn wi pm2.27 con2d ) AABCZDBAGEF $.

  $( Theorem 8 of [Margaris] p. 60.  (Contributed by NM, 5-Aug-1993.)  (Proof
     shortened by Josh Purinton, 29-Dec-2000.) $)
  mth8 $p |- ( ph -> ( -. ps -> -. ( ph -> ps ) ) ) $=
    ( wi pm2.27 con3d ) AABCBABDE $.

  ${
    jc.1 $e |- ( ph -> ps ) $.
    jc.2 $e |- ( ph -> ch ) $.
    $( Inference joining the consequents of two premises.  (Contributed by NM,
       28-Dec-1992.) $)
    jc $p |- ( ph -> -. ( ps -> -. ch ) ) $=
      ( wn wi pm3.2im sylc ) ABCBCFGFDEBCHI $.
  $}

  ${
    impi.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( An importation inference.  (Contributed by NM, 29-Dec-1992.)  (Proof
       shortened by Wolf Lammen, 20-Jul-2013.) $)
    impi $p |- ( -. ( ph -> -. ps ) -> ch ) $=
      ( wn wi con3rr3 con1i ) CABEFABCDGH $.
  $}

  ${
    expi.1 $e |- ( -. ( ph -> -. ps ) -> ch ) $.
    $( An exportation inference.  (Contributed by NM, 29-Dec-1992.)  (Proof
       shortened by Mel L. O'Cat, 28-Nov-2008.) $)
    expi $p |- ( ph -> ( ps -> ch ) ) $=
      ( wn wi pm3.2im syl6 ) ABABEFECABGDH $.
  $}

  $( Simplification.  Similar to Theorem *3.27 (Simp) of [WhiteheadRussell]
     p. 112.  (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Wolf
     Lammen, 13-Nov-2012.) $)
  simprim $p |- ( -. ( ph -> -. ps ) -> ps ) $=
    ( idd impi ) ABBABCD $.

  $( Simplification.  Similar to Theorem *3.26 (Simp) of [WhiteheadRussell]
     p. 112.  (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Wolf
     Lammen, 21-Jul-2012.) $)
  simplim $p |- ( -. ( ph -> ps ) -> ph ) $=
    ( wi pm2.21 con1i ) AABCABDE $.

  $( Theorem *2.5 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 9-Oct-2012.) $)
  pm2.5 $p |- ( -. ( ph -> ps ) -> ( -. ph -> ps ) ) $=
    ( wi wn simplim pm2.24d ) ABCDABABEF $.

  $( Theorem *2.51 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.51 $p |- ( -. ( ph -> ps ) -> ( ph -> -. ps ) ) $=
    ( wi wn ax-1 con3i a1d ) ABCZDBDABHBAEFG $.

  $( Theorem *2.521 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 8-Oct-2012.) $)
  pm2.521 $p |- ( -. ( ph -> ps ) -> ( ps -> ph ) ) $=
    ( wi wn simplim a1d ) ABCDABABEF $.

  $( Theorem *2.52 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 8-Oct-2012.) $)
  pm2.52 $p |- ( -. ( ph -> ps ) -> ( -. ph -> -. ps ) ) $=
    ( wi wn pm2.521 con3d ) ABCDBAABEF $.

  $( Exportation theorem expressed with primitive connectives.  (Contributed by
     NM, 28-Dec-1992.) $)
  expt $p |- ( ( -. ( ph -> -. ps ) -> ch ) -> ( ph -> ( ps -> ch ) ) ) $=
    ( wn wi pm3.2im imim1d com12 ) AABDEDZCEBCEABICABFGH $.

  $( Importation theorem expressed with primitive connectives.  (Contributed by
     NM, 25-Apr-1994.)  (Proof shortened by Wolf Lammen, 20-Jul-2013.) $)
  impt $p |- ( ( ph -> ( ps -> ch ) ) -> ( -. ( ph -> -. ps ) -> ch ) ) $=
    ( wi wn simprim simplim imim1i mpdi ) ABCDZDABEZDEZBCABFLAJAKGHI $.

  ${
    pm2.61d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    pm2.61d.2 $e |- ( ph -> ( -. ps -> ch ) ) $.
    $( Deduction eliminating an antecedent.  (Contributed by NM, 27-Apr-1994.)
       (Proof shortened by Wolf Lammen, 12-Sep-2013.) $)
    pm2.61d $p |- ( ph -> ch ) $=
      ( wn con1d syld pm2.18d ) ACACFBCABCEGDHI $.
  $}

  ${
    pm2.61d1.1 $e |- ( ph -> ( ps -> ch ) ) $.
    pm2.61d1.2 $e |- ( -. ps -> ch ) $.
    $( Inference eliminating an antecedent.  (Contributed by NM,
       15-Jul-2005.) $)
    pm2.61d1 $p |- ( ph -> ch ) $=
      ( wn wi a1i pm2.61d ) ABCDBFCGAEHI $.
  $}

  ${
    pm2.61d2.1 $e |- ( ph -> ( -. ps -> ch ) ) $.
    pm2.61d2.2 $e |- ( ps -> ch ) $.
    $( Inference eliminating an antecedent.  (Contributed by NM,
       18-Aug-1993.) $)
    pm2.61d2 $p |- ( ph -> ch ) $=
      ( wi a1i pm2.61d ) ABCBCFAEGDH $.
  $}

  ${
    ja.1 $e |- ( -. ph -> ch ) $.
    ja.2 $e |- ( ps -> ch ) $.
    $( Inference joining the antecedents of two premises.  (Contributed by NM,
       24-Jan-1993.)  (Proof shortened by Mel L. O'Cat, 19-Feb-2008.) $)
    ja $p |- ( ( ph -> ps ) -> ch ) $=
      ( wi imim2i pm2.61d1 ) ABFACBCAEGDH $.
  $}

  ${
    jad.1 $e |- ( ph -> ( -. ps -> th ) ) $.
    jad.2 $e |- ( ph -> ( ch -> th ) ) $.
    $( Deduction form of ~ ja .  (Contributed by Scott Fenton, 13-Dec-2010.)
       (Proof shortened by Andrew Salmon, 17-Sep-2011.) $)
    jad $p |- ( ph -> ( ( ps -> ch ) -> th ) ) $=
      ( wi wn com12 ja ) BCGADBCADGABHDEIACDFIJI $.
  $}

  $( Elimination of a nested antecedent as a kind of reversal of inference
     ~ ja .  (Contributed by Wolf Lammen, 10-May-2013.) $)
  jarl $p |- ( ( ( ph -> ps ) -> ch ) -> ( -. ph -> ch ) ) $=
    ( wn wi pm2.21 imim1i ) ADABECABFG $.

  ${
    pm2.61i.1 $e |- ( ph -> ps ) $.
    pm2.61i.2 $e |- ( -. ph -> ps ) $.
    $( Inference eliminating an antecedent.  (Contributed by NM, 5-Apr-1994.)
       (Proof shortened by Wolf Lammen, 12-Sep-2013.) $)
    pm2.61i $p |- ps $=
      ( wi id ja ax-mp ) AAEBAFAABDCGH $.
  $}

  ${
    pm2.61ii.1 $e |- ( -. ph -> ( -. ps -> ch ) ) $.
    pm2.61ii.2 $e |- ( ph -> ch ) $.
    pm2.61ii.3 $e |- ( ps -> ch ) $.
    $( Inference eliminating two antecedents.  (Contributed by NM,
       4-Jan-1993.)  (Proof shortened by Josh Purinton, 29-Dec-2000.) $)
    pm2.61ii $p |- ch $=
      ( wn pm2.61d2 pm2.61i ) ACEAGBCDFHI $.
  $}

  ${
    pm2.61nii.1 $e |- ( ph -> ( ps -> ch ) ) $.
    pm2.61nii.2 $e |- ( -. ph -> ch ) $.
    pm2.61nii.3 $e |- ( -. ps -> ch ) $.
    $( Inference eliminating two antecedents.  (Contributed by NM,
       13-Jul-2005.)  (Proof shortened by Andrew Salmon, 25-May-2011.)  (Proof
       shortened by Wolf Lammen, 13-Nov-2012.) $)
    pm2.61nii $p |- ch $=
      ( pm2.61d1 pm2.61i ) ACABCDFGEH $.
  $}

  ${
    pm2.61iii.1 $e |- ( -. ph -> ( -. ps -> ( -. ch -> th ) ) ) $.
    pm2.61iii.2 $e |- ( ph -> th ) $.
    pm2.61iii.3 $e |- ( ps -> th ) $.
    pm2.61iii.4 $e |- ( ch -> th ) $.
    $( Inference eliminating three antecedents.  (Contributed by NM,
       2-Jan-2002.)  (Proof shortened by Wolf Lammen, 22-Sep-2013.) $)
    pm2.61iii $p |- th $=
      ( wn wi a1d pm2.61ii pm2.61i ) CDHABCIZDJEADNFKBDNGKLM $.
  $}

  $( Reductio ad absurdum.  Theorem *2.01 of [WhiteheadRussell] p. 100.  Also
     called the weak Clavius law, and provable in minimal calculus, contrary to
     the Clavius law ~ pm2.18 .  (Contributed by NM, 18-Aug-1993.)  (Proof
     shortened by Mel L. O'Cat, 21-Nov-2008.)  (Proof shortened by Wolf Lammen,
     31-Oct-2012.) $)
  pm2.01 $p |- ( ( ph -> -. ph ) -> -. ph ) $=
    ( wn id ja ) AABZEECZFD $.

  ${
    pm2.01d.1 $e |- ( ph -> ( ps -> -. ps ) ) $.
    $( Deduction based on reductio ad absurdum.  (Contributed by NM,
       18-Aug-1993.)  (Proof shortened by Wolf Lammen, 5-Mar-2013.) $)
    pm2.01d $p |- ( ph -> -. ps ) $=
      ( wn id pm2.61d1 ) ABBDZCGEF $.
  $}

  $( Theorem *2.6 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.6 $p |- ( ( -. ph -> ps ) -> ( ( ph -> ps ) -> ps ) ) $=
    ( wn wi id idd jad ) ACBDZABBHEHBFG $.

  $( Theorem *2.61 of [WhiteheadRussell] p. 107.  Useful for eliminating an
     antecedent.  (Contributed by NM, 4-Jan-1993.)  (Proof shortened by Wolf
     Lammen, 22-Sep-2013.) $)
  pm2.61 $p |- ( ( ph -> ps ) -> ( ( -. ph -> ps ) -> ps ) ) $=
    ( wn wi pm2.6 com12 ) ACBDABDBABEF $.

  $( Theorem *2.65 of [WhiteheadRussell] p. 107.  Proof by contradiction.
     (Contributed by NM, 21-Jun-1993.)  (Proof shortened by Wolf Lammen,
     8-Mar-2013.) $)
  pm2.65 $p |- ( ( ph -> ps ) -> ( ( ph -> -. ps ) -> -. ph ) ) $=
    ( wi wn idd con3 jad ) ABCZABDADZHIEABFG $.

  ${
    pm2.65i.1 $e |- ( ph -> ps ) $.
    pm2.65i.2 $e |- ( ph -> -. ps ) $.
    $( Inference rule for proof by contradiction.  (Contributed by NM,
       18-May-1994.)  (Proof shortened by Wolf Lammen, 11-Sep-2013.) $)
    pm2.65i $p |- -. ph $=
      ( wn con2i con3i pm2.61i ) BAEABDFABCGH $.
  $}

  ${
    pm2.21dd.1 $e |- ( ph -> ps ) $.
    pm2.21dd.2 $e |- ( ph -> -. ps ) $.
    $( A contradiction implies anything.  Deduction from ~ pm2.21 .
       (Contributed by Mario Carneiro, 9-Feb-2017.)  (Proof shortened by Wolf
       Lammen, 22-Jul-2019.) $)
    pm2.21dd $p |- ( ph -> ch ) $=
      ( pm2.65i pm2.21i ) ACABDEFG $.
  $}

  ${
    pm2.65d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    pm2.65d.2 $e |- ( ph -> ( ps -> -. ch ) ) $.
    $( Deduction rule for proof by contradiction.  (Contributed by NM,
       26-Jun-1994.)  (Proof shortened by Wolf Lammen, 26-May-2013.) $)
    pm2.65d $p |- ( ph -> -. ps ) $=
      ( nsyld pm2.01d ) ABABCBEDFG $.
  $}

  ${
    mto.1 $e |- -. ps $.
    mto.2 $e |- ( ph -> ps ) $.
    $( The rule of modus tollens.  The rule says, "if ` ps ` is not true, and
       ` ph ` implies ` ps ` , then ` ph ` must also be not true."  Modus
       tollens is short for "modus tollendo tollens," a Latin phrase that means
       "the mode that by denying denies" - remark in [Sanford] p. 39.  It is
       also called denying the consequent.  Modus tollens is closely related to
       modus ponens ~ ax-mp .  Note that this rule is also valid in
       intuitionistic logic.  Inference associated with ~ con3i .  (Contributed
       by NM, 19-Aug-1993.)  (Proof shortened by Wolf Lammen, 11-Sep-2013.) $)
    mto $p |- -. ph $=
      ( wn a1i pm2.65i ) ABDBEACFG $.
  $}

  ${
    mtod.1 $e |- ( ph -> -. ch ) $.
    mtod.2 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Modus tollens deduction.  (Contributed by NM, 3-Apr-1994.)  (Proof
       shortened by Wolf Lammen, 11-Sep-2013.) $)
    mtod $p |- ( ph -> -. ps ) $=
      ( wn a1d pm2.65d ) ABCEACFBDGH $.
  $}

  ${
    mtoi.1 $e |- -. ch $.
    mtoi.2 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Modus tollens inference.  (Contributed by NM, 5-Jul-1994.)  (Proof
       shortened by Wolf Lammen, 15-Sep-2012.) $)
    mtoi $p |- ( ph -> -. ps ) $=
      ( wn a1i mtod ) ABCCFADGEH $.
  $}

  ${
    mt2.1 $e |- ps $.
    mt2.2 $e |- ( ph -> -. ps ) $.
    $( A rule similar to modus tollens.  Inference associated with ~ con2i .
       (Contributed by NM, 19-Aug-1993.)  (Proof shortened by Wolf Lammen,
       10-Sep-2013.) $)
    mt2 $p |- -. ph $=
      ( a1i pm2.65i ) ABBACEDF $.
  $}

  ${
    mt3.1 $e |- -. ps $.
    mt3.2 $e |- ( -. ph -> ps ) $.
    $( A rule similar to modus tollens.  Inference associated with ~ con1i .
       (Contributed by NM, 18-May-1994.)  (Proof shortened by Wolf Lammen,
       11-Sep-2013.) $)
    mt3 $p |- ph $=
      ( wn mto notnotri ) AAEBCDFG $.
  $}

  $( Peirce's axiom.  This odd-looking theorem is the "difference" between an
     intuitionistic system of propositional calculus and a classical system and
     is not accepted by intuitionists.  When Peirce's axiom is added to an
     intuitionistic system, the system becomes equivalent to our classical
     system ~ ax-1 through ~ ax-3 .  A curious fact about this theorem is that
     it requires ~ ax-3 for its proof even though the result has no negation
     connectives in it.  (Contributed by NM, 29-Dec-1992.)  (Proof shortened by
     Wolf Lammen, 9-Oct-2012.) $)
  peirce $p |- ( ( ( ph -> ps ) -> ph ) -> ph ) $=
    ( wi simplim id ja ) ABCAAABDAEF $.

  $( The Inversion Axiom of the infinite-valued sentential logic (L-infinity)
     of Lukasiewicz.  Using ~ dfor2 , we can see that this essentially
     expresses "disjunction commutes."  Theorem *2.69 of [WhiteheadRussell]
     p. 108.  (Contributed by NM, 12-Aug-2004.) $)
  looinv $p |- ( ( ( ph -> ps ) -> ps ) -> ( ( ps -> ph ) -> ph ) ) $=
    ( wi imim1 peirce syl6 ) ABCZBCBACGACAGBADABEF $.

  $( Theorem used to justify definition of biconditional ~ df-bi .
     (Contributed by NM, 11-May-1999.)  (Proof shortened by Josh Purinton,
     29-Dec-2000.) $)
  bijust $p |- -. ( ( -. ( ( ph -> ps ) -> -. ( ps -> ph ) )
                      -> -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) )
               -> -. ( -. ( ( ph -> ps ) -> -. ( ps -> ph ) )
                      -> -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) ) ) $=
    ( wi wn id pm2.01 mt2 ) ABCBACDCDZHCZIDCIHEIFG $.

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Logical equivalence
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  The definition ~ df-bi in this section is our first definition, which
  introduces and defines the biconditional connective ` <-> ` . We define a wff
  of the form ` ( ph <-> ps ) ` as an abbreviation for
  ` -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) ` .

  Unlike most traditional developments, we have chosen not to have a separate
  symbol such as "Df." to mean "is defined as."  Instead, we will later use the
  biconditional connective for this purpose ( ~ df-or is its first use), as it
  allows us to use logic to manipulate definitions directly.  This greatly
  simplifies many proofs since it eliminates the need for a separate mechanism
  for introducing and eliminating definitions.
$)

  $( Declare the biconditional connective. $)
  $c <-> $. $( Double arrow (read:  'if and only if' or
               'is logically equivalent to') $)

  $( Extend our wff definition to include the biconditional connective. $)
  wb $a wff ( ph <-> ps ) $.

  $( Define the biconditional (logical 'iff').

     The definition ~ df-bi in this section is our first definition, which
     introduces and defines the biconditional connective ` <-> ` .  We define a
     wff of the form ` ( ph <-> ps ) ` as an abbreviation for
     ` -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) ` .

     Unlike most traditional developments, we have chosen not to have a
     separate symbol such as "Df." to mean "is defined as."  Instead, we will
     later use the biconditional connective for this purpose ( ~ df-or is its
     first use), as it allows us to use logic to manipulate definitions
     directly.  This greatly simplifies many proofs since it eliminates the
     need for a separate mechanism for introducing and eliminating
     definitions.  Of course, we cannot use this mechanism to define the
     biconditional itself, since it hasn't been introduced yet.  Instead, we
     use a more general form of definition, described as follows.

     In its most general form, a definition is simply an assertion that
     introduces a new symbol (or a new combination of existing symbols, as in
     ~ df-3an ) that is eliminable and does not strengthen the existing
     language.  The latter requirement means that the set of provable
     statements not containing the new symbol (or new combination) should
     remain exactly the same after the definition is introduced.  Our
     definition of the biconditional may look unusual compared to most
     definitions, but it strictly satisfies these requirements.

     The justification for our definition is that if we mechanically replace
     ` ( ph <-> ps ) ` (the definiendum i.e. the thing being defined) with
     ` -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) ` (the definiens i.e. the
     defining expression) in the definition, the definition becomes the
     previously proved theorem ~ bijust .  It is impossible to use ~ df-bi to
     prove any statement expressed in the original language that can't be
     proved from the original axioms, because if we simply replace each
     instance of ~ df-bi in the proof with the corresponding ~ bijust instance,
     we will end up with a proof from the original axioms.

     Note that from Metamath's point of view, a definition is just another
     axiom - i.e. an assertion we claim to be true - but from our high level
     point of view, we are not strengthening the language.  To indicate this
     fact, we prefix definition labels with "df-" instead of "ax-".  (This
     prefixing is an informal convention that means nothing to the Metamath
     proof verifier; it is just a naming convention for human readability.)

     After we define the constant true ` T. ` ( ~ df-tru ) and the constant
     false ` F. ` ( ~ df-fal ), we will be able to prove these truth table
     values: ` ( ( T. <-> T. ) <-> T. ) ` ( ~ trubitru ),
     ` ( ( T. <-> F. ) <-> F. ) ` ( ~ trubifal ), ` ( ( F. <-> T. ) <-> F. ) `
     ( ~ falbitru ), and ` ( ( F. <-> F. ) <-> T. ) ` ( ~ falbifal ).

     See ~ dfbi1 , ~ dfbi2 , and ~ dfbi3 for theorems suggesting typical
     textbook definitions of ` <-> ` , showing that our definition has the
     properties we expect.  Theorem ~ dfbi1 is particularly useful if we want
     to eliminate ` <-> ` from an expression to convert it to primitives.
     Theorem ~ dfbi shows this definition rewritten in an abbreviated form
     after conjunction is introduced, for easier understanding.

     Contrast with ` \/ ` ( ~ df-or ), ` -> ` ( ~ wi ), ` -/\ ` ( ~ df-nan ),
     and ` \/_ ` ( ~ df-xor ) .  In some sense ` <-> ` returns true if two
     truth values are equal; ` = ` ( ~ df-cleq ) returns true if two classes
     are equal.  (Contributed by NM, 27-Dec-1992.) $)
  df-bi $a |- -. ( ( ( ph <-> ps ) -> -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) )
        -> -. ( -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) -> ( ph <-> ps ) ) ) $.

  $( $j justification 'bijust' for 'df-bi'; $)

  $( Property of the biconditional connective.  (Contributed by NM,
     11-May-1999.) $)
  bi1 $p |- ( ( ph <-> ps ) -> ( ph -> ps ) ) $=
    ( wb wi wn df-bi simplim ax-mp syl ) ABCZABDZBADEZDEZKJMDZMJDEZDENABFNOGHKL
    GI $.

  $( Property of the biconditional connective.  (Contributed by NM,
     11-May-1999.) $)
  bi3 $p |- ( ( ph -> ps ) -> ( ( ps -> ph ) -> ( ph <-> ps ) ) ) $=
    ( wi wb wn df-bi simprim ax-mp expi ) ABCZBACZABDZLJKECEZCZMLCZECEOABFNOGHI
    $.

  ${
    impbii.1 $e |- ( ph -> ps ) $.
    impbii.2 $e |- ( ps -> ph ) $.
    $( Infer an equivalence from an implication and its converse.  (Contributed
       by NM, 29-Dec-1992.) $)
    impbii $p |- ( ph <-> ps ) $=
      ( wi wb bi3 mp2 ) ABEBAEABFCDABGH $.
  $}

  ${
    impbidd.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    impbidd.2 $e |- ( ph -> ( ps -> ( th -> ch ) ) ) $.
    $( Deduce an equivalence from two implications.  (Contributed by Rodolfo
       Medina, 12-Oct-2010.) $)
    impbidd $p |- ( ph -> ( ps -> ( ch <-> th ) ) ) $=
      ( wi wb bi3 syl6c ) ABCDGDCGCDHEFCDIJ $.
  $}

  ${
    impbid21d.1 $e |- ( ps -> ( ch -> th ) ) $.
    impbid21d.2 $e |- ( ph -> ( th -> ch ) ) $.
    $( Deduce an equivalence from two implications.  (Contributed by Wolf
       Lammen, 12-May-2013.) $)
    impbid21d $p |- ( ph -> ( ps -> ( ch <-> th ) ) ) $=
      ( wi a1i a1d impbidd ) ABCDBCDGGAEHADCGBFIJ $.
  $}

  ${
    impbid.1 $e |- ( ph -> ( ps -> ch ) ) $.
    impbid.2 $e |- ( ph -> ( ch -> ps ) ) $.
    $( Deduce an equivalence from two implications.  (Contributed by NM,
       24-Jan-1993.)  Revised to prove it from ~ impbid21d .  (Revised by Wolf
       Lammen, 3-Nov-2012.) $)
    impbid $p |- ( ph -> ( ps <-> ch ) ) $=
      ( wb impbid21d pm2.43i ) ABCFAABCDEGH $.
  $}

  $( Relate the biconditional connective to primitive connectives.  See
     ~ dfbi1ALT for an unusual version proved directly from axioms.
     (Contributed by NM, 29-Dec-1992.) $)
  dfbi1 $p |- ( ( ph <-> ps ) <-> -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) ) $=
    ( wb wi wn df-bi simplim ax-mp bi3 impi impbii ) ABCZABDZBADZEDEZLODZOLDEZD
    EPABFPQGHMNLABIJK $.

  $( Alternate proof of ~ dfbi1 .  This proof, discovered by Gregory Bush on
     8-Mar-2004, has several curious properties.  First, it has only 17 steps
     directly from the axioms and ~ df-bi , compared to over 800 steps were the
     proof of ~ dfbi1 expanded into axioms.  Second, step 2 demands only the
     property of "true"; any axiom (or theorem) could be used.  It might be
     thought, therefore, that it is in some sense redundant, but in fact no
     proof is shorter than this (measured by number of steps).  Third, it
     illustrates how intermediate steps can "blow up" in size even in short
     proofs.  Fourth, the compressed proof is only 182 bytes (or 17 bytes in
     D-proof notation), but the generated web page is over 200kB with
     intermediate steps that are essentially incomprehensible to humans (other
     than Gregory Bush).  If there were an obfuscated code contest for proofs,
     this would be a contender.  This "blowing up" and incomprehensibility of
     the intermediate steps vividly demonstrate the advantages of using many
     layered intermediate theorems, since each theorem is easier to
     understand.  (Contributed by Gregory Bush, 10-Mar-2004.)
     (New usage is discouraged.)  (Proof modification is discouraged.) $)
  dfbi1ALT $p |-
                ( ( ph <-> ps ) <-> -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) ) $=
    ( wch wth wb wi wn df-bi ax-1 ax-mp ax-3 ax-2 ) ABEZABFBAFGFGZFNMFGFGZMNEZA
    BHCDCFFZOPFZCDIRGZQGZFZQRFSPOFZSFZFZUASUBISUCTFZFZUDUAFUEUFTGZUCGZFZUEUHUIM
    NHUHUGIJTUCKJUESIJSUCTLJJRQKJJJ $.

  ${
    biimpi.1 $e |- ( ph <-> ps ) $.
    $( Infer an implication from a logical equivalence.  (Contributed by NM,
       29-Dec-1992.) $)
    biimpi $p |- ( ph -> ps ) $=
      ( wb wi bi1 ax-mp ) ABDABECABFG $.
  $}

  ${
    sylbi.1 $e |- ( ph <-> ps ) $.
    sylbi.2 $e |- ( ps -> ch ) $.
    $( A mixed syllogism inference from a biconditional and an implication.
       Useful for substituting an antecedent with a definition.  (Contributed
       by NM, 3-Jan-1993.) $)
    sylbi $p |- ( ph -> ch ) $=
      ( biimpi syl ) ABCABDFEG $.
  $}

  ${
    sylib.1 $e |- ( ph -> ps ) $.
    sylib.2 $e |- ( ps <-> ch ) $.
    $( A mixed syllogism inference from an implication and a biconditional.
       (Contributed by NM, 3-Jan-1993.) $)
    sylib $p |- ( ph -> ch ) $=
      ( biimpi syl ) ABCDBCEFG $.
  $}

  ${
    sylbb.1 $e |- ( ph <-> ps ) $.
    sylbb.2 $e |- ( ps <-> ch ) $.
    $( A mixed syllogism inference from two biconditionals.  (Contributed by
       BJ, 30-Mar-2019.) $)
    sylbb $p |- ( ph -> ch ) $=
      ( biimpi sylbi ) ABCDBCEFG $.
  $}

  $( Property of the biconditional connective.  (Contributed by NM,
     11-May-1999.)  (Proof shortened by Wolf Lammen, 11-Nov-2012.) $)
  bi2 $p |- ( ( ph <-> ps ) -> ( ps -> ph ) ) $=
    ( wb wi wn dfbi1 simprim sylbi ) ABCABDZBADZEDEJABFIJGH $.

  $( Commutative law for equivalence.  (Contributed by Wolf Lammen,
     10-Nov-2012.) $)
  bicom1 $p |- ( ( ph <-> ps ) -> ( ps <-> ph ) ) $=
    ( wb bi2 bi1 impbid ) ABCBAABDABEF $.

  $( Commutative law for equivalence.  Theorem *4.21 of [WhiteheadRussell]
     p. 117.  (Contributed by NM, 11-May-1993.) $)
  bicom $p |- ( ( ph <-> ps ) <-> ( ps <-> ph ) ) $=
    ( wb bicom1 impbii ) ABCBACABDBADE $.

  ${
    bicomd.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Commute two sides of a biconditional in a deduction.  (Contributed by
       NM, 14-May-1993.) $)
    bicomd $p |- ( ph -> ( ch <-> ps ) ) $=
      ( wb bicom sylib ) ABCECBEDBCFG $.
  $}

  ${
    bicomi.1 $e |- ( ph <-> ps ) $.
    $( Inference from commutative law for logical equivalence.  (Contributed by
       NM, 3-Jan-1993.) $)
    bicomi $p |- ( ps <-> ph ) $=
      ( wb bicom1 ax-mp ) ABDBADCABEF $.
  $}

  ${
    impbid1.1 $e |- ( ph -> ( ps -> ch ) ) $.
    impbid1.2 $e |- ( ch -> ps ) $.
    $( Infer an equivalence from two implications.  (Contributed by NM,
       6-Mar-2007.) $)
    impbid1 $p |- ( ph -> ( ps <-> ch ) ) $=
      ( wi a1i impbid ) ABCDCBFAEGH $.
  $}

  ${
    impbid2.1 $e |- ( ps -> ch ) $.
    impbid2.2 $e |- ( ph -> ( ch -> ps ) ) $.
    $( Infer an equivalence from two implications.  (Contributed by NM,
       6-Mar-2007.)  (Proof shortened by Wolf Lammen, 27-Sep-2013.) $)
    impbid2 $p |- ( ph -> ( ps <-> ch ) ) $=
      ( impbid1 bicomd ) ACBACBEDFG $.
  $}

  ${
    impcon4bid.1 $e |- ( ph -> ( ps -> ch ) ) $.
    impcon4bid.2 $e |- ( ph -> ( -. ps -> -. ch ) ) $.
    $( A variation on ~ impbid with contraposition.  (Contributed by Jeff
       Hankins, 3-Jul-2009.) $)
    impcon4bid $p |- ( ph -> ( ps <-> ch ) ) $=
      ( con4d impbid ) ABCDABCEFG $.
  $}

  ${
    biimpri.1 $e |- ( ph <-> ps ) $.
    $( Infer a converse implication from a logical equivalence.  (Contributed
       by NM, 29-Dec-1992.)  (Proof shortened by Wolf Lammen, 16-Sep-2013.) $)
    biimpri $p |- ( ps -> ph ) $=
      ( bicomi biimpi ) BAABCDE $.
  $}

  ${
    biimpd.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Deduce an implication from a logical equivalence.  (Contributed by NM,
       11-Jan-1993.) $)
    biimpd $p |- ( ph -> ( ps -> ch ) ) $=
      ( wb wi bi1 syl ) ABCEBCFDBCGH $.
  $}

  ${
    mpbi.min $e |- ph $.
    mpbi.maj $e |- ( ph <-> ps ) $.
    $( An inference from a biconditional, related to modus ponens.
       (Contributed by NM, 11-May-1993.) $)
    mpbi $p |- ps $=
      ( biimpi ax-mp ) ABCABDEF $.
  $}

  ${
    mpbir.min $e |- ps $.
    mpbir.maj $e |- ( ph <-> ps ) $.
    $( An inference from a biconditional, related to modus ponens.
       (Contributed by NM, 28-Dec-1992.) $)
    mpbir $p |- ph $=
      ( biimpri ax-mp ) BACABDEF $.
  $}

  ${
    mpbid.min $e |- ( ph -> ps ) $.
    mpbid.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( A deduction from a biconditional, related to modus ponens.  (Contributed
       by NM, 21-Jun-1993.) $)
    mpbid $p |- ( ph -> ch ) $=
      ( biimpd mpd ) ABCDABCEFG $.
  $}

  ${
    mpbii.min $e |- ps $.
    mpbii.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( An inference from a nested biconditional, related to modus ponens.
       (Contributed by NM, 16-May-1993.)  (Proof shortened by Wolf Lammen,
       25-Oct-2012.) $)
    mpbii $p |- ( ph -> ch ) $=
      ( a1i mpbid ) ABCBADFEG $.
  $}

  ${
    sylibr.1 $e |- ( ph -> ps ) $.
    sylibr.2 $e |- ( ch <-> ps ) $.
    $( A mixed syllogism inference from an implication and a biconditional.
       Useful for substituting a consequent with a definition.  (Contributed by
       NM, 3-Jan-1993.) $)
    sylibr $p |- ( ph -> ch ) $=
      ( biimpri syl ) ABCDCBEFG $.
  $}

  ${
    sylbir.1 $e |- ( ps <-> ph ) $.
    sylbir.2 $e |- ( ps -> ch ) $.
    $( A mixed syllogism inference from a biconditional and an implication.
       (Contributed by NM, 3-Jan-1993.) $)
    sylbir $p |- ( ph -> ch ) $=
      ( biimpri syl ) ABCBADFEG $.
  $}

  ${
    sylibd.1 $e |- ( ph -> ( ps -> ch ) ) $.
    sylibd.2 $e |- ( ph -> ( ch <-> th ) ) $.
    $( A syllogism deduction.  (Contributed by NM, 3-Aug-1994.) $)
    sylibd $p |- ( ph -> ( ps -> th ) ) $=
      ( biimpd syld ) ABCDEACDFGH $.
  $}

  ${
    sylbid.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    sylbid.2 $e |- ( ph -> ( ch -> th ) ) $.
    $( A syllogism deduction.  (Contributed by NM, 3-Aug-1994.) $)
    sylbid $p |- ( ph -> ( ps -> th ) ) $=
      ( biimpd syld ) ABCDABCEGFH $.
  $}

  ${
    mpbidi.min $e |- ( th -> ( ph -> ps ) ) $.
    mpbidi.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( A deduction from a biconditional, related to modus ponens.  (Contributed
       by NM, 9-Aug-1994.) $)
    mpbidi $p |- ( th -> ( ph -> ch ) ) $=
      ( biimpd sylcom ) DABCEABCFGH $.
  $}

  ${
    syl5bi.1 $e |- ( ph <-> ps ) $.
    syl5bi.2 $e |- ( ch -> ( ps -> th ) ) $.
    $( A mixed syllogism inference from a nested implication and a
       biconditional.  Useful for substituting an embedded antecedent with a
       definition.  (Contributed by NM, 12-Jan-1993.) $)
    syl5bi $p |- ( ch -> ( ph -> th ) ) $=
      ( biimpi syl5 ) ABCDABEGFH $.
  $}

  ${
    syl5bir.1 $e |- ( ps <-> ph ) $.
    syl5bir.2 $e |- ( ch -> ( ps -> th ) ) $.
    $( A mixed syllogism inference from a nested implication and a
       biconditional.  (Contributed by NM, 21-Jun-1993.) $)
    syl5bir $p |- ( ch -> ( ph -> th ) ) $=
      ( biimpri syl5 ) ABCDBAEGFH $.
  $}

  ${
    syl5ib.1 $e |- ( ph -> ps ) $.
    syl5ib.2 $e |- ( ch -> ( ps <-> th ) ) $.
    $( A mixed syllogism inference.  (Contributed by NM, 12-Jan-1993.) $)
    syl5ib $p |- ( ch -> ( ph -> th ) ) $=
      ( biimpd syl5 ) ABCDECBDFGH $.

    $( A mixed syllogism inference.  (Contributed by NM, 19-Jun-2007.) $)
    syl5ibcom $p |- ( ph -> ( ch -> th ) ) $=
      ( syl5ib com12 ) CADABCDEFGH $.
  $}

  ${
    syl5ibr.1 $e |- ( ph -> th ) $.
    syl5ibr.2 $e |- ( ch -> ( ps <-> th ) ) $.
    $( A mixed syllogism inference.  (Contributed by NM, 3-Apr-1994.) $)
    syl5ibr $p |- ( ch -> ( ph -> ps ) ) $=
      ( bicomd syl5ib ) ADCBECBDFGH $.

    $( A mixed syllogism inference.  (Contributed by NM, 20-Jun-2007.) $)
    syl5ibrcom $p |- ( ph -> ( ch -> ps ) ) $=
      ( syl5ibr com12 ) CABABCDEFGH $.
  $}

  ${
    biimprd.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Deduce a converse implication from a logical equivalence.  (Contributed
       by NM, 11-Jan-1993.)  (Proof shortened by Wolf Lammen, 22-Sep-2013.) $)
    biimprd $p |- ( ph -> ( ch -> ps ) ) $=
      ( id syl5ibr ) CBACCEDF $.
  $}

  ${
    biimpcd.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Deduce a commuted implication from a logical equivalence.  (Contributed
       by NM, 3-May-1994.)  (Proof shortened by Wolf Lammen, 22-Sep-2013.) $)
    biimpcd $p |- ( ps -> ( ph -> ch ) ) $=
      ( id syl5ibcom ) BBACBEDF $.

    $( Deduce a converse commuted implication from a logical equivalence.
       (Contributed by NM, 3-May-1994.)  (Proof shortened by Wolf Lammen,
       20-Dec-2013.) $)
    biimprcd $p |- ( ch -> ( ph -> ps ) ) $=
      ( id syl5ibrcom ) CBACCEDF $.
  $}

  ${
    syl6ib.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl6ib.2 $e |- ( ch <-> th ) $.
    $( A mixed syllogism inference from a nested implication and a
       biconditional.  (Contributed by NM, 21-Jun-1993.) $)
    syl6ib $p |- ( ph -> ( ps -> th ) ) $=
      ( biimpi syl6 ) ABCDECDFGH $.
  $}

  ${
    syl6ibr.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl6ibr.2 $e |- ( th <-> ch ) $.
    $( A mixed syllogism inference from a nested implication and a
       biconditional.  Useful for substituting an embedded consequent with a
       definition.  (Contributed by NM, 10-Jan-1993.) $)
    syl6ibr $p |- ( ph -> ( ps -> th ) ) $=
      ( biimpri syl6 ) ABCDEDCFGH $.
  $}

  ${
    syl6bi.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    syl6bi.2 $e |- ( ch -> th ) $.
    $( A mixed syllogism inference.  (Contributed by NM, 2-Jan-1994.) $)
    syl6bi $p |- ( ph -> ( ps -> th ) ) $=
      ( biimpd syl6 ) ABCDABCEGFH $.
  $}

  ${
    syl6bir.1 $e |- ( ph -> ( ch <-> ps ) ) $.
    syl6bir.2 $e |- ( ch -> th ) $.
    $( A mixed syllogism inference.  (Contributed by NM, 18-May-1994.) $)
    syl6bir $p |- ( ph -> ( ps -> th ) ) $=
      ( biimprd syl6 ) ABCDACBEGFH $.
  $}

  ${
    syl7bi.1 $e |- ( ph <-> ps ) $.
    syl7bi.2 $e |- ( ch -> ( th -> ( ps -> ta ) ) ) $.
    $( A mixed syllogism inference from a doubly nested implication and a
       biconditional.  (Contributed by NM, 14-May-1993.) $)
    syl7bi $p |- ( ch -> ( th -> ( ph -> ta ) ) ) $=
      ( biimpi syl7 ) ABCDEABFHGI $.
  $}

  ${
    syl8ib.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    syl8ib.2 $e |- ( th <-> ta ) $.
    $( A syllogism rule of inference.  The second premise is used to replace
       the consequent of the first premise.  (Contributed by NM,
       1-Aug-1994.) $)
    syl8ib $p |- ( ph -> ( ps -> ( ch -> ta ) ) ) $=
      ( biimpi syl8 ) ABCDEFDEGHI $.
  $}

  ${
    mpbird.min $e |- ( ph -> ch ) $.
    mpbird.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( A deduction from a biconditional, related to modus ponens.  (Contributed
       by NM, 5-Aug-1993.) $)
    mpbird $p |- ( ph -> ps ) $=
      ( biimprd mpd ) ACBDABCEFG $.
  $}

  ${
    mpbiri.min $e |- ch $.
    mpbiri.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( An inference from a nested biconditional, related to modus ponens.
       (Contributed by NM, 21-Jun-1993.)  (Proof shortened by Wolf Lammen,
       25-Oct-2012.) $)
    mpbiri $p |- ( ph -> ps ) $=
      ( a1i mpbird ) ABCCADFEG $.
  $}

  ${
    sylibrd.1 $e |- ( ph -> ( ps -> ch ) ) $.
    sylibrd.2 $e |- ( ph -> ( th <-> ch ) ) $.
    $( A syllogism deduction.  (Contributed by NM, 3-Aug-1994.) $)
    sylibrd $p |- ( ph -> ( ps -> th ) ) $=
      ( biimprd syld ) ABCDEADCFGH $.
  $}

  ${
    sylbird.1 $e |- ( ph -> ( ch <-> ps ) ) $.
    sylbird.2 $e |- ( ph -> ( ch -> th ) ) $.
    $( A syllogism deduction.  (Contributed by NM, 3-Aug-1994.) $)
    sylbird $p |- ( ph -> ( ps -> th ) ) $=
      ( biimprd syld ) ABCDACBEGFH $.
  $}

  $( Principle of identity for logical equivalence.  Theorem *4.2 of
     [WhiteheadRussell] p. 117.  This is part of Frege's eighth axiom per
     Proposition 54 of [Frege1879] p. 50; see also ~ eqid .  (Contributed by
     NM, 2-Jun-1993.) $)
  biid $p |- ( ph <-> ph ) $=
    ( id impbii ) AAABZDC $.

  $( Principle of identity with antecedent.  (Contributed by NM,
     25-Nov-1995.) $)
  biidd $p |- ( ph -> ( ps <-> ps ) ) $=
    ( wb biid a1i ) BBCABDE $.

  $( Two propositions are equivalent if they are both true.  Closed form of
     ~ 2th .  Equivalent to a ~ bi1 -like version of the xor-connective.  This
     theorem stays true, no matter how you permute its operands.  This is
     evident from its sharper version
     ` ( ph <-> ( ps <-> ( ph <-> ps ) ) ) ` .  (Contributed by Wolf Lammen,
     12-May-2013.) $)
  pm5.1im $p |- ( ph -> ( ps -> ( ph <-> ps ) ) ) $=
    ( ax-1 impbid21d ) ABABBACABCD $.

  ${
    2th.1 $e |- ph $.
    2th.2 $e |- ps $.
    $( Two truths are equivalent.  (Contributed by NM, 18-Aug-1993.) $)
    2th $p |- ( ph <-> ps ) $=
      ( a1i impbii ) ABBADEABCEF $.
  $}

  ${
    2thd.1 $e |- ( ph -> ps ) $.
    2thd.2 $e |- ( ph -> ch ) $.
    $( Two truths are equivalent (deduction rule).  (Contributed by NM,
       3-Jun-2012.) $)
    2thd $p |- ( ph -> ( ps <-> ch ) ) $=
      ( wb pm5.1im sylc ) ABCBCFDEBCGH $.
  $}

  ${
    ibi.1 $e |- ( ph -> ( ph <-> ps ) ) $.
    $( Inference that converts a biconditional implied by one of its arguments,
       into an implication.  (Contributed by NM, 17-Oct-2003.) $)
    ibi $p |- ( ph -> ps ) $=
      ( biimpd pm2.43i ) ABAABCDE $.
  $}

  ${
    ibir.1 $e |- ( ph -> ( ps <-> ph ) ) $.
    $( Inference that converts a biconditional implied by one of its arguments,
       into an implication.  (Contributed by NM, 22-Jul-2004.) $)
    ibir $p |- ( ph -> ps ) $=
      ( bicomd ibi ) ABABACDE $.
  $}

  ${
    ibd.1 $e |- ( ph -> ( ps -> ( ps <-> ch ) ) ) $.
    $( Deduction that converts a biconditional implied by one of its arguments,
       into an implication.  (Contributed by NM, 26-Jun-2004.) $)
    ibd $p |- ( ph -> ( ps -> ch ) ) $=
      ( wb bi1 syli ) BABCECDBCFG $.
  $}

  $( Distribution of implication over biconditional.  Theorem *5.74 of
     [WhiteheadRussell] p. 126.  (Contributed by NM, 1-Aug-1994.)  (Proof
     shortened by Wolf Lammen, 11-Apr-2013.) $)
  pm5.74 $p |- ( ( ph -> ( ps <-> ch ) ) <->
               ( ( ph -> ps ) <-> ( ph -> ch ) ) ) $=
    ( wb wi bi1 imim3i bi2 impbid pm2.86d impbidd impbii ) ABCDZEZABEZACEZDZNOP
    MBCABCFGMCBABCHGIQABCQABCOPFJQACBOPHJKL $.

  ${
    pm5.74i.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Distribution of implication over biconditional (inference rule).
       (Contributed by NM, 1-Aug-1994.) $)
    pm5.74i $p |- ( ( ph -> ps ) <-> ( ph -> ch ) ) $=
      ( wb wi pm5.74 mpbi ) ABCEFABFACFEDABCGH $.
  $}

  ${
    pm5.74ri.1 $e |- ( ( ph -> ps ) <-> ( ph -> ch ) ) $.
    $( Distribution of implication over biconditional (reverse inference
       rule).  (Contributed by NM, 1-Aug-1994.) $)
    pm5.74ri $p |- ( ph -> ( ps <-> ch ) ) $=
      ( wb wi pm5.74 mpbir ) ABCEFABFACFEDABCGH $.
  $}

  ${
    pm5.74d.1 $e |- ( ph -> ( ps -> ( ch <-> th ) ) ) $.
    $( Distribution of implication over biconditional (deduction rule).
       (Contributed by NM, 21-Mar-1996.) $)
    pm5.74d $p |- ( ph -> ( ( ps -> ch ) <-> ( ps -> th ) ) ) $=
      ( wb wi pm5.74 sylib ) ABCDFGBCGBDGFEBCDHI $.
  $}

  ${
    pm5.74rd.1 $e |- ( ph -> ( ( ps -> ch ) <-> ( ps -> th ) ) ) $.
    $( Distribution of implication over biconditional (deduction rule).
       (Contributed by NM, 19-Mar-1997.) $)
    pm5.74rd $p |- ( ph -> ( ps -> ( ch <-> th ) ) ) $=
      ( wi wb pm5.74 sylibr ) ABCFBDFGBCDGFEBCDHI $.
  $}

  ${
    bitri.1 $e |- ( ph <-> ps ) $.
    bitri.2 $e |- ( ps <-> ch ) $.
    $( An inference from transitive law for logical equivalence.  (Contributed
       by NM, 3-Jan-1993.)  (Proof shortened by Wolf Lammen, 13-Oct-2012.) $)
    bitri $p |- ( ph <-> ch ) $=
      ( sylbb biimpri sylibr impbii ) ACABCDEFCBABCEGDHI $.
  $}

  ${
    bitr2i.1 $e |- ( ph <-> ps ) $.
    bitr2i.2 $e |- ( ps <-> ch ) $.
    $( An inference from transitive law for logical equivalence.  (Contributed
       by NM, 12-Mar-1993.) $)
    bitr2i $p |- ( ch <-> ph ) $=
      ( bitri bicomi ) ACABCDEFG $.
  $}

  ${
    bitr3i.1 $e |- ( ps <-> ph ) $.
    bitr3i.2 $e |- ( ps <-> ch ) $.
    $( An inference from transitive law for logical equivalence.  (Contributed
       by NM, 2-Jun-1993.) $)
    bitr3i $p |- ( ph <-> ch ) $=
      ( bicomi bitri ) ABCBADFEG $.
  $}

  ${
    bitr4i.1 $e |- ( ph <-> ps ) $.
    bitr4i.2 $e |- ( ch <-> ps ) $.
    $( An inference from transitive law for logical equivalence.  (Contributed
       by NM, 3-Jan-1993.) $)
    bitr4i $p |- ( ph <-> ch ) $=
      ( bicomi bitri ) ABCDCBEFG $.
  $}

  $( Register '<->' as an equality for its type (wff). $)
  $( $j
    equality 'wb' from 'biid' 'bicomi' 'bitri';
    definition 'dfbi1' for 'wb';
  $)

  ${
    bitrd.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    bitrd.2 $e |- ( ph -> ( ch <-> th ) ) $.
    $( Deduction form of ~ bitri .  (Contributed by NM, 12-Mar-1993.)  (Proof
       shortened by Wolf Lammen, 14-Apr-2013.) $)
    bitrd $p |- ( ph -> ( ps <-> th ) ) $=
      ( wi pm5.74i bitri pm5.74ri ) ABDABGACGADGABCEHACDFHIJ $.
  $}

  ${
    bitr2d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    bitr2d.2 $e |- ( ph -> ( ch <-> th ) ) $.
    $( Deduction form of ~ bitr2i .  (Contributed by NM, 9-Jun-2004.) $)
    bitr2d $p |- ( ph -> ( th <-> ps ) ) $=
      ( bitrd bicomd ) ABDABCDEFGH $.
  $}

  ${
    bitr3d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    bitr3d.2 $e |- ( ph -> ( ps <-> th ) ) $.
    $( Deduction form of ~ bitr3i .  (Contributed by NM, 14-May-1993.) $)
    bitr3d $p |- ( ph -> ( ch <-> th ) ) $=
      ( bicomd bitrd ) ACBDABCEGFH $.
  $}

  ${
    bitr4d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    bitr4d.2 $e |- ( ph -> ( th <-> ch ) ) $.
    $( Deduction form of ~ bitr4i .  (Contributed by NM, 30-Jun-1993.) $)
    bitr4d $p |- ( ph -> ( ps <-> th ) ) $=
      ( bicomd bitrd ) ABCDEADCFGH $.
  $}

  ${
    syl5bb.1 $e |- ( ph <-> ps ) $.
    syl5bb.2 $e |- ( ch -> ( ps <-> th ) ) $.
    $( A syllogism inference from two biconditionals.  (Contributed by NM,
       12-Mar-1993.) $)
    syl5bb $p |- ( ch -> ( ph <-> th ) ) $=
      ( wb a1i bitrd ) CABDABGCEHFI $.
  $}

  ${
    syl5rbb.1 $e |- ( ph <-> ps ) $.
    syl5rbb.2 $e |- ( ch -> ( ps <-> th ) ) $.
    $( A syllogism inference from two biconditionals.  (Contributed by NM,
       1-Aug-1993.) $)
    syl5rbb $p |- ( ch -> ( th <-> ph ) ) $=
      ( syl5bb bicomd ) CADABCDEFGH $.
  $}

  ${
    syl5bbr.1 $e |- ( ps <-> ph ) $.
    syl5bbr.2 $e |- ( ch -> ( ps <-> th ) ) $.
    $( A syllogism inference from two biconditionals.  (Contributed by NM,
       5-Aug-1993.) $)
    syl5bbr $p |- ( ch -> ( ph <-> th ) ) $=
      ( bicomi syl5bb ) ABCDBAEGFH $.
  $}

  ${
    syl5rbbr.1 $e |- ( ps <-> ph ) $.
    syl5rbbr.2 $e |- ( ch -> ( ps <-> th ) ) $.
    $( A syllogism inference from two biconditionals.  (Contributed by NM,
       25-Nov-1994.) $)
    syl5rbbr $p |- ( ch -> ( th <-> ph ) ) $=
      ( bicomi syl5rbb ) ABCDBAEGFH $.
  $}

  ${
    syl6bb.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    syl6bb.2 $e |- ( ch <-> th ) $.
    $( A syllogism inference from two biconditionals.  (Contributed by NM,
       12-Mar-1993.) $)
    syl6bb $p |- ( ph -> ( ps <-> th ) ) $=
      ( wb a1i bitrd ) ABCDECDGAFHI $.
  $}

  ${
    syl6rbb.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    syl6rbb.2 $e |- ( ch <-> th ) $.
    $( A syllogism inference from two biconditionals.  (Contributed by NM,
       5-Aug-1993.) $)
    syl6rbb $p |- ( ph -> ( th <-> ps ) ) $=
      ( syl6bb bicomd ) ABDABCDEFGH $.
  $}

  ${
    syl6bbr.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    syl6bbr.2 $e |- ( th <-> ch ) $.
    $( A syllogism inference from two biconditionals.  (Contributed by NM,
       12-Mar-1993.) $)
    syl6bbr $p |- ( ph -> ( ps <-> th ) ) $=
      ( bicomi syl6bb ) ABCDEDCFGH $.
  $}

  ${
    syl6rbbr.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    syl6rbbr.2 $e |- ( th <-> ch ) $.
    $( A syllogism inference from two biconditionals.  (Contributed by NM,
       25-Nov-1994.) $)
    syl6rbbr $p |- ( ph -> ( th <-> ps ) ) $=
      ( bicomi syl6rbb ) ABCDEDCFGH $.
  $}

  ${
    3imtr3.1 $e |- ( ph -> ps ) $.
    3imtr3.2 $e |- ( ph <-> ch ) $.
    3imtr3.3 $e |- ( ps <-> th ) $.
    $( A mixed syllogism inference, useful for removing a definition from both
       sides of an implication.  (Contributed by NM, 10-Aug-1994.) $)
    3imtr3i $p |- ( ch -> th ) $=
      ( sylbir sylib ) CBDCABFEHGI $.
  $}

  ${
    3imtr4.1 $e |- ( ph -> ps ) $.
    3imtr4.2 $e |- ( ch <-> ph ) $.
    3imtr4.3 $e |- ( th <-> ps ) $.
    $( A mixed syllogism inference, useful for applying a definition to both
       sides of an implication.  (Contributed by NM, 3-Jan-1993.) $)
    3imtr4i $p |- ( ch -> th ) $=
      ( sylbi sylibr ) CBDCABFEHGI $.
  $}

  ${
    3imtr3d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3imtr3d.2 $e |- ( ph -> ( ps <-> th ) ) $.
    3imtr3d.3 $e |- ( ph -> ( ch <-> ta ) ) $.
    $( More general version of ~ 3imtr3i .  Useful for converting conditional
       definitions in a formula.  (Contributed by NM, 8-Apr-1996.) $)
    3imtr3d $p |- ( ph -> ( th -> ta ) ) $=
      ( sylibd sylbird ) ADBEGABCEFHIJ $.
  $}

  ${
    3imtr4d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3imtr4d.2 $e |- ( ph -> ( th <-> ps ) ) $.
    3imtr4d.3 $e |- ( ph -> ( ta <-> ch ) ) $.
    $( More general version of ~ 3imtr4i .  Useful for converting conditional
       definitions in a formula.  (Contributed by NM, 26-Oct-1995.) $)
    3imtr4d $p |- ( ph -> ( th -> ta ) ) $=
      ( sylibrd sylbid ) ADBEGABCEFHIJ $.
  $}

  ${
    3imtr3g.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3imtr3g.2 $e |- ( ps <-> th ) $.
    3imtr3g.3 $e |- ( ch <-> ta ) $.
    $( More general version of ~ 3imtr3i .  Useful for converting definitions
       in a formula.  (Contributed by NM, 20-May-1996.)  (Proof shortened by
       Wolf Lammen, 20-Dec-2013.) $)
    3imtr3g $p |- ( ph -> ( th -> ta ) ) $=
      ( syl5bir syl6ib ) ADCEDBACGFIHJ $.
  $}

  ${
    3imtr4g.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3imtr4g.2 $e |- ( th <-> ps ) $.
    3imtr4g.3 $e |- ( ta <-> ch ) $.
    $( More general version of ~ 3imtr4i .  Useful for converting definitions
       in a formula.  (Contributed by NM, 20-May-1996.)  (Proof shortened by
       Wolf Lammen, 20-Dec-2013.) $)
    3imtr4g $p |- ( ph -> ( th -> ta ) ) $=
      ( syl5bi syl6ibr ) ADCEDBACGFIHJ $.
  $}

  ${
    3bitri.1 $e |- ( ph <-> ps ) $.
    3bitri.2 $e |- ( ps <-> ch ) $.
    3bitri.3 $e |- ( ch <-> th ) $.
    $( A chained inference from transitive law for logical equivalence.
       (Contributed by NM, 3-Jan-1993.) $)
    3bitri $p |- ( ph <-> th ) $=
      ( bitri ) ABDEBCDFGHH $.

    $( A chained inference from transitive law for logical equivalence.
       (Contributed by NM, 4-Aug-2006.) $)
    3bitrri $p |- ( th <-> ph ) $=
      ( bitr2i bitr3i ) DCAGABCEFHI $.
  $}

  ${
    3bitr2i.1 $e |- ( ph <-> ps ) $.
    3bitr2i.2 $e |- ( ch <-> ps ) $.
    3bitr2i.3 $e |- ( ch <-> th ) $.
    $( A chained inference from transitive law for logical equivalence.
       (Contributed by NM, 4-Aug-2006.) $)
    3bitr2i $p |- ( ph <-> th ) $=
      ( bitr4i bitri ) ACDABCEFHGI $.

    $( A chained inference from transitive law for logical equivalence.
       (Contributed by NM, 4-Aug-2006.) $)
    3bitr2ri $p |- ( th <-> ph ) $=
      ( bitr4i bitr2i ) ACDABCEFHGI $.
  $}

  ${
    3bitr3i.1 $e |- ( ph <-> ps ) $.
    3bitr3i.2 $e |- ( ph <-> ch ) $.
    3bitr3i.3 $e |- ( ps <-> th ) $.
    $( A chained inference from transitive law for logical equivalence.
       (Contributed by NM, 19-Aug-1993.) $)
    3bitr3i $p |- ( ch <-> th ) $=
      ( bitr3i bitri ) CBDCABFEHGI $.

    $( A chained inference from transitive law for logical equivalence.
       (Contributed by NM, 21-Jun-1993.) $)
    3bitr3ri $p |- ( th <-> ch ) $=
      ( bitr3i ) DBCGBACEFHH $.
  $}

  ${
    3bitr4i.1 $e |- ( ph <-> ps ) $.
    3bitr4i.2 $e |- ( ch <-> ph ) $.
    3bitr4i.3 $e |- ( th <-> ps ) $.
    $( A chained inference from transitive law for logical equivalence.  This
       inference is frequently used to apply a definition to both sides of a
       logical equivalence.  (Contributed by NM, 3-Jan-1993.) $)
    3bitr4i $p |- ( ch <-> th ) $=
      ( bitr4i bitri ) CADFABDEGHI $.

    $( A chained inference from transitive law for logical equivalence.
       (Contributed by NM, 2-Sep-1995.) $)
    3bitr4ri $p |- ( th <-> ch ) $=
      ( bitr4i bitr2i ) CADFABDEGHI $.
  $}

  ${
    3bitrd.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    3bitrd.2 $e |- ( ph -> ( ch <-> th ) ) $.
    3bitrd.3 $e |- ( ph -> ( th <-> ta ) ) $.
    $( Deduction from transitivity of biconditional.  (Contributed by NM,
       13-Aug-1999.) $)
    3bitrd $p |- ( ph -> ( ps <-> ta ) ) $=
      ( bitrd ) ABDEABCDFGIHI $.

    $( Deduction from transitivity of biconditional.  (Contributed by NM,
       4-Aug-2006.) $)
    3bitrrd $p |- ( ph -> ( ta <-> ps ) ) $=
      ( bitr2d bitr3d ) ADEBHABCDFGIJ $.
  $}

  ${
    3bitr2d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    3bitr2d.2 $e |- ( ph -> ( th <-> ch ) ) $.
    3bitr2d.3 $e |- ( ph -> ( th <-> ta ) ) $.
    $( Deduction from transitivity of biconditional.  (Contributed by NM,
       4-Aug-2006.) $)
    3bitr2d $p |- ( ph -> ( ps <-> ta ) ) $=
      ( bitr4d bitrd ) ABDEABCDFGIHJ $.

    $( Deduction from transitivity of biconditional.  (Contributed by NM,
       4-Aug-2006.) $)
    3bitr2rd $p |- ( ph -> ( ta <-> ps ) ) $=
      ( bitr4d bitr2d ) ABDEABCDFGIHJ $.
  $}

  ${
    3bitr3d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    3bitr3d.2 $e |- ( ph -> ( ps <-> th ) ) $.
    3bitr3d.3 $e |- ( ph -> ( ch <-> ta ) ) $.
    $( Deduction from transitivity of biconditional.  Useful for converting
       conditional definitions in a formula.  (Contributed by NM,
       24-Apr-1996.) $)
    3bitr3d $p |- ( ph -> ( th <-> ta ) ) $=
      ( bitr3d bitrd ) ADCEABDCGFIHJ $.

    $( Deduction from transitivity of biconditional.  (Contributed by NM,
       4-Aug-2006.) $)
    3bitr3rd $p |- ( ph -> ( ta <-> th ) ) $=
      ( bitr3d ) ACEDHABCDFGII $.
  $}

  ${
    3bitr4d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    3bitr4d.2 $e |- ( ph -> ( th <-> ps ) ) $.
    3bitr4d.3 $e |- ( ph -> ( ta <-> ch ) ) $.
    $( Deduction from transitivity of biconditional.  Useful for converting
       conditional definitions in a formula.  (Contributed by NM,
       18-Oct-1995.) $)
    3bitr4d $p |- ( ph -> ( th <-> ta ) ) $=
      ( bitr4d bitrd ) ADBEGABCEFHIJ $.

    $( Deduction from transitivity of biconditional.  (Contributed by NM,
       4-Aug-2006.) $)
    3bitr4rd $p |- ( ph -> ( ta <-> th ) ) $=
      ( bitr4d ) AEBDAECBHFIGI $.
  $}

  ${
    3bitr3g.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    3bitr3g.2 $e |- ( ps <-> th ) $.
    3bitr3g.3 $e |- ( ch <-> ta ) $.
    $( More general version of ~ 3bitr3i .  Useful for converting definitions
       in a formula.  (Contributed by NM, 4-Jun-1995.) $)
    3bitr3g $p |- ( ph -> ( th <-> ta ) ) $=
      ( syl5bbr syl6bb ) ADCEDBACGFIHJ $.
  $}

  ${
    3bitr4g.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    3bitr4g.2 $e |- ( th <-> ps ) $.
    3bitr4g.3 $e |- ( ta <-> ch ) $.
    $( More general version of ~ 3bitr4i .  Useful for converting definitions
       in a formula.  (Contributed by NM, 11-May-1993.) $)
    3bitr4g $p |- ( ph -> ( th <-> ta ) ) $=
      ( syl5bb syl6bbr ) ADCEDBACGFIHJ $.
  $}

  ${
    bi3ant.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Obsolete theorem as of 27-May-2019.  (Contributed by Wolf Lammen,
       14-May-2013.)  (Proof modification is discouraged.)
       (New usage is discouraged.) $)
    bi3antOLD $p |- ( ( ( th -> ta ) -> ph )
        -> ( ( ( ta -> th ) -> ps ) -> ( ( th <-> ta ) -> ch ) ) ) $=
      ( wi wb bi1 imim1i bi2 imim3i syl2im ) DEGZAGDEHZAGEDGZBGOBGOCGONADEIJOPB
      DEKJABCOFLM $.
  $}

  $( Obsolete theorem as of 27-May-2019.  (Contributed by Wolf Lammen,
     14-May-2013.)  (Proof modification is discouraged.)
     (New usage is discouraged.) $)
  bisymOLD $p |- ( ( ( ph -> ps ) -> ( ch -> th ) ) -> ( ( ( ps -> ph )
      -> ( th -> ch ) ) -> ( ( ph <-> ps ) -> ( ch <-> th ) ) ) ) $=
    ( wi wb bi3 bi3antOLD ) CDEDCECDFABCDGH $.

  $( Double negation.  Theorem *4.13 of [WhiteheadRussell] p. 117.
     (Contributed by NM, 3-Jan-1993.) $)
  notnot $p |- ( ph <-> -. -. ph ) $=
    ( wn notnot1 notnot2 impbii ) AABBACADE $.

  $( Contraposition.  Theorem *4.1 of [WhiteheadRussell] p. 116.  (Contributed
     by NM, 11-May-1993.) $)
  con34b $p |- ( ( ph -> ps ) <-> ( -. ps -> -. ph ) ) $=
    ( wi wn con3 ax-3 impbii ) ABCBDADCABEBAFG $.

  ${
    con4bid.1 $e |- ( ph -> ( -. ps <-> -. ch ) ) $.
    $( A contraposition deduction.  (Contributed by NM, 21-May-1994.) $)
    con4bid $p |- ( ph -> ( ps <-> ch ) ) $=
      ( wn biimprd con4d biimpd impcon4bid ) ABCACBABEZCEZDFGAJKDHI $.
  $}

  ${
    notbid.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Deduction negating both sides of a logical equivalence.  (Contributed by
       NM, 21-May-1994.) $)
    notbid $p |- ( ph -> ( -. ps <-> -. ch ) ) $=
      ( wn notnot 3bitr3g con4bid ) ABEZCEZABCIEJEDBFCFGH $.
  $}

  $( Contraposition.  Theorem *4.11 of [WhiteheadRussell] p. 117.  (Contributed
     by NM, 21-May-1994.)  (Proof shortened by Wolf Lammen, 12-Jun-2013.) $)
  notbi $p |- ( ( ph <-> ps ) <-> ( -. ph <-> -. ps ) ) $=
    ( wb wn id notbid con4bid impbii ) ABCZADBDCZIABIEFJABJEGH $.

  ${
    notbii.1 $e |- ( ph <-> ps ) $.
    $( Negate both sides of a logical equivalence.  (Contributed by NM,
       3-Jan-1993.)  (Proof shortened by Wolf Lammen, 19-May-2013.) $)
    notbii $p |- ( -. ph <-> -. ps ) $=
      ( wb wn notbi mpbi ) ABDAEBEDCABFG $.

    $( Theorem notbii is the congruence law for negation. $)
    $( $j congruence 'notbii'; $)
  $}

  ${
    con4bii.1 $e |- ( -. ph <-> -. ps ) $.
    $( A contraposition inference.  (Contributed by NM, 21-May-1994.) $)
    con4bii $p |- ( ph <-> ps ) $=
      ( wb wn notbi mpbir ) ABDAEBEDCABFG $.
  $}

  ${
    mtbi.1 $e |- -. ph $.
    mtbi.2 $e |- ( ph <-> ps ) $.
    $( An inference from a biconditional, related to modus tollens.
       (Contributed by NM, 15-Nov-1994.)  (Proof shortened by Wolf Lammen,
       25-Oct-2012.) $)
    mtbi $p |- -. ps $=
      ( biimpri mto ) BACABDEF $.
  $}

  ${
    mtbir.1 $e |- -. ps $.
    mtbir.2 $e |- ( ph <-> ps ) $.
    $( An inference from a biconditional, related to modus tollens.
       (Contributed by NM, 15-Nov-1994.)  (Proof shortened by Wolf Lammen,
       14-Oct-2012.) $)
    mtbir $p |- -. ph $=
      ( bicomi mtbi ) BACABDEF $.
  $}

  ${
    mtbid.min $e |- ( ph -> -. ps ) $.
    mtbid.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( A deduction from a biconditional, similar to modus tollens.
       (Contributed by NM, 26-Nov-1995.) $)
    mtbid $p |- ( ph -> -. ch ) $=
      ( biimprd mtod ) ACBDABCEFG $.
  $}

  ${
    mtbird.min $e |- ( ph -> -. ch ) $.
    mtbird.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( A deduction from a biconditional, similar to modus tollens.
       (Contributed by NM, 10-May-1994.) $)
    mtbird $p |- ( ph -> -. ps ) $=
      ( biimpd mtod ) ABCDABCEFG $.
  $}

  ${
    mtbii.min $e |- -. ps $.
    mtbii.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( An inference from a biconditional, similar to modus tollens.
       (Contributed by NM, 27-Nov-1995.) $)
    mtbii $p |- ( ph -> -. ch ) $=
      ( biimprd mtoi ) ACBDABCEFG $.
  $}

  ${
    mtbiri.min $e |- -. ch $.
    mtbiri.maj $e |- ( ph -> ( ps <-> ch ) ) $.
    $( An inference from a biconditional, similar to modus tollens.
       (Contributed by NM, 24-Aug-1995.) $)
    mtbiri $p |- ( ph -> -. ps ) $=
      ( biimpd mtoi ) ABCDABCEFG $.
  $}

  ${
    sylnib.1 $e |- ( ph -> -. ps ) $.
    sylnib.2 $e |- ( ps <-> ch ) $.
    $( A mixed syllogism inference from an implication and a biconditional.
       (Contributed by Wolf Lammen, 16-Dec-2013.) $)
    sylnib $p |- ( ph -> -. ch ) $=
      ( wb a1i mtbid ) ABCDBCFAEGH $.
  $}

  ${
    sylnibr.1 $e |- ( ph -> -. ps ) $.
    sylnibr.2 $e |- ( ch <-> ps ) $.
    $( A mixed syllogism inference from an implication and a biconditional.
       Useful for substituting a consequent with a definition.  (Contributed by
       Wolf Lammen, 16-Dec-2013.) $)
    sylnibr $p |- ( ph -> -. ch ) $=
      ( bicomi sylnib ) ABCDCBEFG $.
  $}

  ${
    sylnbi.1 $e |- ( ph <-> ps ) $.
    sylnbi.2 $e |- ( -. ps -> ch ) $.
    $( A mixed syllogism inference from a biconditional and an implication.
       Useful for substituting an antecedent with a definition.  (Contributed
       by Wolf Lammen, 16-Dec-2013.) $)
    sylnbi $p |- ( -. ph -> ch ) $=
      ( wn notbii sylbi ) AFBFCABDGEH $.
  $}

  ${
    sylnbir.1 $e |- ( ps <-> ph ) $.
    sylnbir.2 $e |- ( -. ps -> ch ) $.
    $( A mixed syllogism inference from a biconditional and an implication.
       (Contributed by Wolf Lammen, 16-Dec-2013.) $)
    sylnbir $p |- ( -. ph -> ch ) $=
      ( bicomi sylnbi ) ABCBADFEG $.
  $}

  ${
    xchnxbi.1 $e |- ( -. ph <-> ps ) $.
    xchnxbi.2 $e |- ( ph <-> ch ) $.
    $( Replacement of a subexpression by an equivalent one.  (Contributed by
       Wolf Lammen, 27-Sep-2014.) $)
    xchnxbi $p |- ( -. ch <-> ps ) $=
      ( wn notbii bitr3i ) CFAFBACEGDH $.
  $}

  ${
    xchnxbir.1 $e |- ( -. ph <-> ps ) $.
    xchnxbir.2 $e |- ( ch <-> ph ) $.
    $( Replacement of a subexpression by an equivalent one.  (Contributed by
       Wolf Lammen, 27-Sep-2014.) $)
    xchnxbir $p |- ( -. ch <-> ps ) $=
      ( bicomi xchnxbi ) ABCDCAEFG $.
  $}

  ${
    xchbinx.1 $e |- ( ph <-> -. ps ) $.
    xchbinx.2 $e |- ( ps <-> ch ) $.
    $( Replacement of a subexpression by an equivalent one.  (Contributed by
       Wolf Lammen, 27-Sep-2014.) $)
    xchbinx $p |- ( ph <-> -. ch ) $=
      ( wn notbii bitri ) ABFCFDBCEGH $.
  $}

  ${
    xchbinxr.1 $e |- ( ph <-> -. ps ) $.
    xchbinxr.2 $e |- ( ch <-> ps ) $.
    $( Replacement of a subexpression by an equivalent one.  (Contributed by
       Wolf Lammen, 27-Sep-2014.) $)
    xchbinxr $p |- ( ph <-> -. ch ) $=
      ( bicomi xchbinx ) ABCDCBEFG $.
  $}

  ${
    imbi2i.1 $e |- ( ph <-> ps ) $.
    $( Introduce an antecedent to both sides of a logical equivalence.  This
       and the next three rules are useful for building up wff's around a
       definition, in order to make use of the definition.  (Contributed by NM,
       3-Jan-1993.)  (Proof shortened by Wolf Lammen, 6-Feb-2013.) $)
    imbi2i $p |- ( ( ch -> ph ) <-> ( ch -> ps ) ) $=
      ( wb a1i pm5.74i ) CABABECDFG $.
  $}

  ${
    bibi2i.1 $e |- ( ph <-> ps ) $.
    $( Inference adding a biconditional to the left in an equivalence.
       (Contributed by NM, 26-May-1993.)  (Proof shortened by Andrew Salmon,
       7-May-2011.)  (Proof shortened by Wolf Lammen, 16-May-2013.) $)
    bibi2i $p |- ( ( ch <-> ph ) <-> ( ch <-> ps ) ) $=
      ( wb id syl6bb syl6bbr impbii ) CAEZCBEZJCABJFDGKCBAKFDHI $.

    $( Inference adding a biconditional to the right in an equivalence.
       (Contributed by NM, 26-May-1993.) $)
    bibi1i $p |- ( ( ph <-> ch ) <-> ( ps <-> ch ) ) $=
      ( wb bicom bibi2i 3bitri ) ACECAECBEBCEACFABCDGCBFH $.

    ${
      bibi12i.2 $e |- ( ch <-> th ) $.
      $( The equivalence of two equivalences.  (Contributed by NM,
         26-May-1993.) $)
      bibi12i $p |- ( ( ph <-> ch ) <-> ( ps <-> th ) ) $=
        ( wb bibi2i bibi1i bitri ) ACGADGBDGCDAFHABDEIJ $.
    $}
  $}

  ${
    imbid.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Deduction adding an antecedent to both sides of a logical equivalence.
       (Contributed by NM, 11-May-1993.) $)
    imbi2d $p |- ( ph -> ( ( th -> ps ) <-> ( th -> ch ) ) ) $=
      ( wb a1d pm5.74d ) ADBCABCFDEGH $.

    $( Deduction adding a consequent to both sides of a logical equivalence.
       (Contributed by NM, 11-May-1993.)  (Proof shortened by Wolf Lammen,
       17-Sep-2013.) $)
    imbi1d $p |- ( ph -> ( ( ps -> th ) <-> ( ch -> th ) ) ) $=
      ( wi biimprd imim1d biimpd impbid ) ABDFCDFACBDABCEGHABCDABCEIHJ $.

    $( Deduction adding a biconditional to the left in an equivalence.
       (Contributed by NM, 11-May-1993.)  (Proof shortened by Wolf Lammen,
       19-May-2013.) $)
    bibi2d $p |- ( ph -> ( ( th <-> ps ) <-> ( th <-> ch ) ) ) $=
      ( wb wi pm5.74i bibi2i pm5.74 3bitr4i pm5.74ri ) ADBFZDCFZADGZABGZFOACGZF
      AMGANGPQOABCEHIADBJADCJKL $.

    $( Deduction adding a biconditional to the right in an equivalence.
       (Contributed by NM, 11-May-1993.) $)
    bibi1d $p |- ( ph -> ( ( ps <-> th ) <-> ( ch <-> th ) ) ) $=
      ( wb bibi2d bicom 3bitr4g ) ADBFDCFBDFCDFABCDEGBDHCDHI $.
  $}

  ${
    imbi12d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    imbi12d.2 $e |- ( ph -> ( th <-> ta ) ) $.
    $( Deduction joining two equivalences to form equivalence of implications.
       (Contributed by NM, 16-May-1993.) $)
    imbi12d $p |- ( ph -> ( ( ps -> th ) <-> ( ch -> ta ) ) ) $=
      ( wi imbi1d imbi2d bitrd ) ABDHCDHCEHABCDFIADECGJK $.

    $( Deduction joining two equivalences to form equivalence of
       biconditionals.  (Contributed by NM, 26-May-1993.) $)
    bibi12d $p |- ( ph -> ( ( ps <-> th ) <-> ( ch <-> ta ) ) ) $=
      ( wb bibi1d bibi2d bitrd ) ABDHCDHCEHABCDFIADECGJK $.
  $}

  $( Closed form of ~ imbi12i .  Was automatically derived from its "Virtual
     Deduction" version and Metamath's "minimize" command.  (Contributed by
     Alan Sare, 18-Mar-2012.) $)
  imbi12 $p |- ( ( ph <-> ps ) ->
                    ( ( ch <-> th ) -> ( ( ph -> ch ) <-> ( ps -> th ) ) ) ) $=
    ( wb wi wn simplim simprim imbi12d expi ) ABEZCDEZACFBDFELMGZFGABCDLNHLMIJK
    $.

  $( Theorem *4.84 of [WhiteheadRussell] p. 122.  (Contributed by NM,
     3-Jan-2005.) $)
  imbi1 $p |- ( ( ph <-> ps ) -> ( ( ph -> ch ) <-> ( ps -> ch ) ) ) $=
    ( wb id imbi1d ) ABDZABCGEF $.

  $( Theorem *4.85 of [WhiteheadRussell] p. 122.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 19-May-2013.) $)
  imbi2 $p |- ( ( ph <-> ps ) -> ( ( ch -> ph ) <-> ( ch -> ps ) ) ) $=
    ( wb id imbi2d ) ABDZABCGEF $.

  ${
    imbi1i.1 $e |- ( ph <-> ps ) $.
    $( Introduce a consequent to both sides of a logical equivalence.
       (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Wolf Lammen,
       17-Sep-2013.) $)
    imbi1i $p |- ( ( ph -> ch ) <-> ( ps -> ch ) ) $=
      ( wb wi imbi1 ax-mp ) ABEACFBCFEDABCGH $.
  $}

  ${
    imbi12i.1 $e |- ( ph <-> ps ) $.
    imbi12i.2 $e |- ( ch <-> th ) $.
    $( Join two logical equivalences to form equivalence of implications.
       (Contributed by NM, 1-Aug-1993.) $)
    imbi12i $p |- ( ( ph -> ch ) <-> ( ps -> th ) ) $=
      ( wb wi imbi12 mp2 ) ABGCDGACHBDHGEFABCDIJ $.

    $( Theorem imbi12i is the congruence law for implication. $)
    $( $j congruence 'imbi12i'; $)
  $}

  $( Theorem *4.86 of [WhiteheadRussell] p. 122.  (Contributed by NM,
     3-Jan-2005.) $)
  bibi1 $p |- ( ( ph <-> ps ) -> ( ( ph <-> ch ) <-> ( ps <-> ch ) ) ) $=
    ( wb id bibi1d ) ABDZABCGEF $.

  $( Contraposition.  Theorem *4.12 of [WhiteheadRussell] p. 117.  (Contributed
     by NM, 15-Apr-1995.)  (Proof shortened by Wolf Lammen, 3-Jan-2013.) $)
  con2bi $p |- ( ( ph <-> -. ps ) <-> ( ps <-> -. ph ) ) $=
    ( wn wb notbi notnot bibi2i bicom 3bitr2i ) ABCZDACZJCZDKBDBKDAJEBLKBFGKBHI
    $.

  ${
    con2bid.1 $e |- ( ph -> ( ps <-> -. ch ) ) $.
    $( A contraposition deduction.  (Contributed by NM, 15-Apr-1995.) $)
    con2bid $p |- ( ph -> ( ch <-> -. ps ) ) $=
      ( wn wb con2bi sylibr ) ABCEFCBEFDCBGH $.
  $}

  ${
    con1bid.1 $e |- ( ph -> ( -. ps <-> ch ) ) $.
    $( A contraposition deduction.  (Contributed by NM, 9-Oct-1999.) $)
    con1bid $p |- ( ph -> ( -. ch <-> ps ) ) $=
      ( wn bicomd con2bid ) ABCEACBABECDFGF $.
  $}

  ${
    con1bii.1 $e |- ( -. ph <-> ps ) $.
    $( A contraposition inference.  (Contributed by NM, 12-Mar-1993.)  (Proof
       shortened by Wolf Lammen, 13-Oct-2012.) $)
    con1bii $p |- ( -. ps <-> ph ) $=
      ( wn notnot xchbinx bicomi ) ABDAADBAECFG $.
  $}

  ${
    con2bii.1 $e |- ( ph <-> -. ps ) $.
    $( A contraposition inference.  (Contributed by NM, 12-Mar-1993.) $)
    con2bii $p |- ( ps <-> -. ph ) $=
      ( wn bicomi con1bii ) ADBBAABDCEFE $.
  $}

  $( Contraposition.  Bidirectional version of ~ con1 .  (Contributed by NM,
     3-Jan-1993.) $)
  con1b $p |- ( ( -. ph -> ps ) <-> ( -. ps -> ph ) ) $=
    ( wn wi con1 impbii ) ACBDBCADABEBAEF $.

  $( Contraposition.  Bidirectional version of ~ con2 .  (Contributed by NM,
     12-Mar-1993.) $)
  con2b $p |- ( ( ph -> -. ps ) <-> ( ps -> -. ph ) ) $=
    ( wn wi con2 impbii ) ABCDBACDABEBAEF $.

  $( A wff is equivalent to itself with true antecedent.  (Contributed by NM,
     28-Jan-1996.) $)
  biimt $p |- ( ph -> ( ps <-> ( ph -> ps ) ) ) $=
    ( wi ax-1 pm2.27 impbid2 ) ABABCBADABEF $.

  $( Theorem *5.5 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.5 $p |- ( ph -> ( ( ph -> ps ) <-> ps ) ) $=
    ( wi biimt bicomd ) ABABCABDE $.

  ${
    a1bi.1 $e |- ph $.
    $( Inference rule introducing a theorem as an antecedent.  (Contributed by
       NM, 5-Aug-1993.)  (Proof shortened by Wolf Lammen, 11-Nov-2012.) $)
    a1bi $p |- ( ps <-> ( ph -> ps ) ) $=
      ( wi wb biimt ax-mp ) ABABDECABFG $.
  $}

  ${
    mt2bi.1 $e |- ph $.
    $( A false consequent falsifies an antecedent.  (Contributed by NM,
       19-Aug-1993.)  (Proof shortened by Wolf Lammen, 12-Nov-2012.) $)
    mt2bi $p |- ( -. ps <-> ( ps -> -. ph ) ) $=
      ( wn wi a1bi con2b bitri ) BDZAIEBADEAICFABGH $.
  $}

  $( Modus-tollens-like theorem.  (Contributed by NM, 7-Apr-2001.)  (Proof
     shortened by Wolf Lammen, 12-Nov-2012.) $)
  mtt $p |- ( -. ph -> ( -. ps <-> ( ps -> ph ) ) ) $=
    ( wn wi biimt con34b syl6bbr ) ACZBCZHIDBADHIEBAFG $.

  $( If a proposition is false, then implying it is equivalent to being false.
     One of four theorems that can be used to simplify an implication
     ` ( ph -> ps ) ` , the other ones being ~ ax-1 (true consequent), ~ pm2.21
     (false antecedent), ~ pm5.5 (true antecedent).  (Contributed by Mario
     Carneiro, 26-Apr-2019.)  (Proof shortened by Wolf Lammen, 26-May-2019.) $)
  imnot $p |- ( -. ps -> ( ( ph -> ps ) <-> -. ph ) ) $=
    ( wn wi mtt bicomd ) BCACABDBAEF $.

  $( Theorem *5.501 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.501 $p |- ( ph -> ( ps <-> ( ph <-> ps ) ) ) $=
    ( wb pm5.1im bi1 com12 impbid ) ABABCZABDHABABEFG $.

  $( Implication in terms of implication and biconditional.  (Contributed by
     NM, 31-Mar-1994.)  (Proof shortened by Wolf Lammen, 24-Jan-2013.) $)
  ibib $p |- ( ( ph -> ps ) <-> ( ph -> ( ph <-> ps ) ) ) $=
    ( wb pm5.501 pm5.74i ) ABABCABDE $.

  $( Implication in terms of implication and biconditional.  (Contributed by
     NM, 29-Apr-2005.)  (Proof shortened by Wolf Lammen, 21-Dec-2013.) $)
  ibibr $p |- ( ( ph -> ps ) <-> ( ph -> ( ps <-> ph ) ) ) $=
    ( wb pm5.501 bicom syl6bb pm5.74i ) ABBACZABABCHABDABEFG $.

  ${
    tbt.1 $e |- ph $.
    $( A wff is equivalent to its equivalence with a truth.  (Contributed by
       NM, 18-Aug-1993.)  (Proof shortened by Andrew Salmon, 13-May-2011.) $)
    tbt $p |- ( ps <-> ( ps <-> ph ) ) $=
      ( wb ibibr pm5.74ri ax-mp ) ABBADZDCABHABEFG $.
  $}

  $( The negation of a wff is equivalent to the wff's equivalence to
     falsehood.  (Contributed by Juha Arpiainen, 19-Jan-2006.)  (Proof
     shortened by Wolf Lammen, 28-Jan-2013.) $)
  nbn2 $p |- ( -. ph -> ( -. ps <-> ( ph <-> ps ) ) ) $=
    ( wn wb pm5.501 notbi syl6bbr ) ACZBCZHIDABDHIEABFG $.

  $( Transfer negation via an equivalence.  (Contributed by NM, 3-Oct-2007.)
     (Proof shortened by Wolf Lammen, 28-Jan-2013.) $)
  bibif $p |- ( -. ps -> ( ( ph <-> ps ) <-> -. ph ) ) $=
    ( wn wb nbn2 bicom syl6rbb ) BCACBADABDBAEBAFG $.

  ${
    nbn.1 $e |- -. ph $.
    $( The negation of a wff is equivalent to the wff's equivalence to
       falsehood.  (Contributed by NM, 21-Jun-1993.)  (Proof shortened by Wolf
       Lammen, 3-Oct-2013.) $)
    nbn $p |- ( -. ps <-> ( ps <-> ph ) ) $=
      ( wb wn bibif ax-mp bicomi ) BADZBEZAEIJDCBAFGH $.
  $}

  ${
    nbn3.1 $e |- ph $.
    $( Transfer falsehood via equivalence.  (Contributed by NM,
       11-Sep-2006.) $)
    nbn3 $p |- ( -. ps <-> ( ps <-> -. ph ) ) $=
      ( wn notnoti nbn ) ADBACEF $.
  $}

  $( Two propositions are equivalent if they are both false.  Closed form of
     ~ 2false .  Equivalent to a ~ bi2 -like version of the xor-connective.
     (Contributed by Wolf Lammen, 13-May-2013.) $)
  pm5.21im $p |- ( -. ph -> ( -. ps -> ( ph <-> ps ) ) ) $=
    ( wn wb nbn2 biimpd ) ACBCABDABEF $.

  ${
    2false.1 $e |- -. ph $.
    2false.2 $e |- -. ps $.
    $( Two falsehoods are equivalent.  (Contributed by NM, 4-Apr-2005.)  (Proof
       shortened by Wolf Lammen, 19-May-2013.) $)
    2false $p |- ( ph <-> ps ) $=
      ( wn 2th con4bii ) ABAEBECDFG $.
  $}

  ${
    2falsed.1 $e |- ( ph -> -. ps ) $.
    2falsed.2 $e |- ( ph -> -. ch ) $.
    $( Two falsehoods are equivalent (deduction rule).  (Contributed by NM,
       11-Oct-2013.) $)
    2falsed $p |- ( ph -> ( ps <-> ch ) ) $=
      ( pm2.21d impbid ) ABCABCDFACBEFG $.
  $}

  ${
    pm5.21ni.1 $e |- ( ph -> ps ) $.
    pm5.21ni.2 $e |- ( ch -> ps ) $.
    $( Two propositions implying a false one are equivalent.  (Contributed by
       NM, 16-Feb-1996.)  (Proof shortened by Wolf Lammen, 19-May-2013.) $)
    pm5.21ni $p |- ( -. ps -> ( ph <-> ch ) ) $=
      ( wn con3i 2falsed ) BFACABDGCBEGH $.

    ${
      pm5.21nii.3 $e |- ( ps -> ( ph <-> ch ) ) $.
      $( Eliminate an antecedent implied by each side of a biconditional.
         (Contributed by NM, 21-May-1999.) $)
      pm5.21nii $p |- ( ph <-> ch ) $=
        ( wb pm5.21ni pm2.61i ) BACGFABCDEHI $.
    $}
  $}

  ${
    pm5.21ndd.1 $e |- ( ph -> ( ch -> ps ) ) $.
    pm5.21ndd.2 $e |- ( ph -> ( th -> ps ) ) $.
    pm5.21ndd.3 $e |- ( ph -> ( ps -> ( ch <-> th ) ) ) $.
    $( Eliminate an antecedent implied by each side of a biconditional,
       deduction version.  (Contributed by Paul Chapman, 21-Nov-2012.)  (Proof
       shortened by Wolf Lammen, 6-Oct-2013.) $)
    pm5.21ndd $p |- ( ph -> ( ch <-> th ) ) $=
      ( wb wn con3d pm5.21im syl6c pm2.61d ) ABCDHZGABICIDINACBEJADBFJCDKLM $.
  $}

  ${
    bija.1 $e |- ( ph -> ( ps -> ch ) ) $.
    bija.2 $e |- ( -. ph -> ( -. ps -> ch ) ) $.
    $( Combine antecedents into a single biconditional.  This inference,
       reminiscent of ~ ja , is reversible:  The hypotheses can be deduced from
       the conclusion alone (see ~ pm5.1im and ~ pm5.21im ).  (Contributed by
       Wolf Lammen, 13-May-2013.) $)
    bija $p |- ( ( ph <-> ps ) -> ch ) $=
      ( wb bi2 syli wn bi1 con3d pm2.61d ) ABFZBCBMACABGDHBIMAICMABABJKEHL $.
  $}

  $( Theorem *5.18 of [WhiteheadRussell] p. 124.  This theorem says that
     logical equivalence is the same as negated "exclusive-or."  (Contributed
     by NM, 28-Jun-2002.)  (Proof shortened by Andrew Salmon, 20-Jun-2011.)
     (Proof shortened by Wolf Lammen, 15-Oct-2013.) $)
  pm5.18 $p |- ( ( ph <-> ps ) <-> -. ( ph <-> -. ps ) ) $=
    ( wb wn pm5.501 con1bid bitr2d nbn2 pm2.61i ) AABCZABDZCZDZCAMBJABLAKEFABEG
    ADZMKJNKLAKHFABHGI $.

  $( Two ways to express "exclusive or."  (Contributed by NM, 1-Jan-2006.) $)
  xor3 $p |- ( -. ( ph <-> ps ) <-> ( ph <-> -. ps ) ) $=
    ( wn wb pm5.18 con2bii bicomi ) ABCDZABDZCIHABEFG $.

  $( Move negation outside of biconditional.  Compare Theorem *5.18 of
     [WhiteheadRussell] p. 124.  (Contributed by NM, 27-Jun-2002.)  (Proof
     shortened by Wolf Lammen, 20-Sep-2013.) $)
  nbbn $p |- ( ( -. ph <-> ps ) <-> -. ( ph <-> ps ) ) $=
    ( wb wn xor3 con2bi bicom 3bitrri ) ABCDABDCBADZCIBCABEABFBIGH $.

  $( Associative law for the biconditional.  An axiom of system DS in Vladimir
     Lifschitz, "On calculational proofs", Annals of Pure and Applied Logic,
     113:207-224, 2002,
     ~ http://www.cs.utexas.edu/users/ai-lab/pub-view.php?PubID=26805 .
     Interestingly, this law was not included in _Principia Mathematica_ but
     was apparently first noted by Jan Lukasiewicz circa 1923.  (Contributed by
     NM, 8-Jan-2005.)  (Proof shortened by Juha Arpiainen, 19-Jan-2006.)
     (Proof shortened by Wolf Lammen, 21-Sep-2013.) $)
  biass $p |- ( ( ( ph <-> ps ) <-> ch ) <-> ( ph <-> ( ps <-> ch ) ) ) $=
    ( wb pm5.501 bibi1d bitr3d wn nbbn nbn2 syl5bbr pm2.61i ) AABDZCDZABCDZDZDA
    ONPABMCABEFAOEGAHZOHZNPRBHZCDQNBCIQSMCABJFKAOJGL $.

  $( Theorem *5.19 of [WhiteheadRussell] p. 124.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.19 $p |- -. ( ph <-> -. ph ) $=
    ( wb wn biid pm5.18 mpbi ) AABAACBCADAAEF $.

  $( Logical equivalence of commuted antecedents.  Part of Theorem *4.87 of
     [WhiteheadRussell] p. 122.  (Contributed by NM, 11-May-1993.) $)
  bi2.04 $p |- ( ( ph -> ( ps -> ch ) ) <-> ( ps -> ( ph -> ch ) ) ) $=
    ( wi pm2.04 impbii ) ABCDDBACDDABCEBACEF $.

  $( Antecedent absorption implication.  Theorem *5.4 of [WhiteheadRussell]
     p. 125.  (Contributed by NM, 5-Aug-1993.) $)
  pm5.4 $p |- ( ( ph -> ( ph -> ps ) ) <-> ( ph -> ps ) ) $=
    ( wi pm2.43 ax-1 impbii ) AABCZCGABDGAEF $.

  $( Distributive law for implication.  Compare Theorem *5.41 of
     [WhiteheadRussell] p. 125.  (Contributed by NM, 5-Aug-1993.) $)
  imdi $p |- ( ( ph -> ( ps -> ch ) ) <->
               ( ( ph -> ps ) -> ( ph -> ch ) ) ) $=
    ( wi ax-2 pm2.86 impbii ) ABCDDABDACDDABCEABCFG $.

  $( Theorem *5.41 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 12-Oct-2012.) $)
  pm5.41 $p |- ( ( ( ph -> ps ) -> ( ph -> ch ) ) <->
                ( ph -> ( ps -> ch ) ) ) $=
    ( wi imdi bicomi ) ABCDDABDACDDABCEF $.

  $( Theorem *4.8 of [WhiteheadRussell] p. 122.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.8 $p |- ( ( ph -> -. ph ) <-> -. ph ) $=
    ( wn wi pm2.01 ax-1 impbii ) AABZCGADGAEF $.

  $( Theorem *4.81 of [WhiteheadRussell] p. 122.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.81 $p |- ( ( -. ph -> ph ) <-> ph ) $=
    ( wn wi pm2.18 pm2.24 impbii ) ABACAADAAEF $.

  $( Simplify an implication between two implications when the antecedent of
     the first is a consequence of the antecedent of the second.  The reverse
     form is useful in producing the successor step in induction proofs.
     (Contributed by Paul Chapman, 22-Jun-2011.)  (Proof shortened by Wolf
     Lammen, 14-Sep-2013.) $)
  imim21b $p |- ( ( ps -> ph ) ->
           ( ( ( ph -> ch ) -> ( ps -> th ) ) <-> ( ps -> ( ch -> th ) ) ) ) $=
    ( wi bi2.04 wb pm5.5 imbi1d imim2i pm5.74d syl5bb ) ACEZBDEEBMDEZEBAEZBCDEZ
    EMBDFOBNPANPGBAMCDACHIJKL $.


$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Logical disjunction and conjunction
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  Here we define disjunction (logical 'or') ` \/ ` ( ~ df-or ) and conjunction
  (logical 'and') ` /\ ` ( ~ df-an ). We also define various rules for
  simplifying and applying them, e.g., ~ olc , ~ orc , and ~ orcom .

$)

  $( Declare connectives for disjunction ('or') and conjunction ('and'). $)
  $c \/ $. $( Vee (read:  'or') $)
  $c /\ $. $( Wedge (read:  'and') $)

  $( Extend wff definition to include disjunction ('or'). $)
  wo $a wff ( ph \/ ps ) $.
  $( Extend wff definition to include conjunction ('and'). $)
  wa $a wff ( ph /\ ps ) $.

  $( Define disjunction (logical 'or').  Definition of [Margaris] p. 49.  When
     the left operand, right operand, or both are true, the result is true;
     when both sides are false, the result is false.  For example, it is true
     that ` ( 2 = 3 \/ 4 = 4 ) ` ( ~ ex-or ).  After we define the constant
     true ` T. ` ( ~ df-tru ) and the constant false ` F. ` ( ~ df-fal ), we
     will be able to prove these truth table values:
     ` ( ( T. \/ T. ) <-> T. ) ` ( ~ truortru ), ` ( ( T. \/ F. ) <-> T. ) `
     ( ~ truorfal ), ` ( ( F. \/ T. ) <-> T. ) ` ( ~ falortru ), and
     ` ( ( F. \/ F. ) <-> F. ) ` ( ~ falorfal ).

     This is our first use of the biconditional connective in a definition; we
     use the biconditional connective in place of the traditional "<=def=>",
     which means the same thing, except that we can manipulate the
     biconditional connective directly in proofs rather than having to rely on
     an informal definition substitution rule.  Note that if we mechanically
     substitute ` ( -. ph -> ps ) ` for ` ( ph \/ ps ) ` , we end up with an
     instance of previously proved theorem ~ biid .  This is the justification
     for the definition, along with the fact that it introduces a new symbol
     ` \/ ` .  Contrast with ` /\ ` ( ~ df-an ), ` -> ` ( ~ wi ), ` -/\ `
     ( ~ df-nan ), and ` \/_ ` ( ~ df-xor ) .  (Contributed by NM,
     27-Dec-1992.) $)
  df-or $a |- ( ( ph \/ ps ) <-> ( -. ph -> ps ) ) $.

  $( Define conjunction (logical 'and').  Definition of [Margaris] p. 49.  When
     both the left and right operand are true, the result is true; when either
     is false, the result is false.  For example, it is true that
     ` ( 2 = 2 /\ 3 = 3 ) ` .  After we define the constant true ` T. `
     ( ~ df-tru ) and the constant false ` F. ` ( ~ df-fal ), we will be able
     to prove these truth table values: ` ( ( T. /\ T. ) <-> T. ) `
     ( ~ truantru ), ` ( ( T. /\ F. ) <-> F. ) ` ( ~ truanfal ),
     ` ( ( F. /\ T. ) <-> F. ) ` ( ~ falantru ), and
     ` ( ( F. /\ F. ) <-> F. ) ` ( ~ falanfal ).

     Contrast with ` \/ ` ( ~ df-or ), ` -> ` ( ~ wi ), ` -/\ ` ( ~ df-nan ),
     and ` \/_ ` ( ~ df-xor ) .  (Contributed by NM, 5-Jan-1993.) $)
  df-an $a |- ( ( ph /\ ps ) <-> -. ( ph -> -. ps ) ) $.

  $( Theorem *4.64 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.64 $p |- ( ( -. ph -> ps ) <-> ( ph \/ ps ) ) $=
    ( wo wn wi df-or bicomi ) ABCADBEABFG $.

  $( Theorem *2.53 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.53 $p |- ( ( ph \/ ps ) -> ( -. ph -> ps ) ) $=
    ( wo wn wi df-or biimpi ) ABCADBEABFG $.

  $( Theorem *2.54 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.54 $p |- ( ( -. ph -> ps ) -> ( ph \/ ps ) ) $=
    ( wo wn wi df-or biimpri ) ABCADBEABFG $.

  ${
    ori.1 $e |- ( ph \/ ps ) $.
    $( Infer implication from disjunction.  (Contributed by NM,
       11-Jun-1994.) $)
    ori $p |- ( -. ph -> ps ) $=
      ( wo wn wi df-or mpbi ) ABDAEBFCABGH $.
  $}

  ${
    orri.1 $e |- ( -. ph -> ps ) $.
    $( Infer disjunction from implication.  (Contributed by NM,
       11-Jun-1994.) $)
    orri $p |- ( ph \/ ps ) $=
      ( wo wn wi df-or mpbir ) ABDAEBFCABGH $.
  $}

  ${
    ord.1 $e |- ( ph -> ( ps \/ ch ) ) $.
    $( Deduce implication from disjunction.  (Contributed by NM,
       18-May-1994.) $)
    ord $p |- ( ph -> ( -. ps -> ch ) ) $=
      ( wo wn wi df-or sylib ) ABCEBFCGDBCHI $.
  $}

  ${
    orrd.1 $e |- ( ph -> ( -. ps -> ch ) ) $.
    $( Deduce disjunction from implication.  (Contributed by NM,
       27-Nov-1995.) $)
    orrd $p |- ( ph -> ( ps \/ ch ) ) $=
      ( wn wi wo pm2.54 syl ) ABECFBCGDBCHI $.
  $}

  ${
    jaoi.1 $e |- ( ph -> ps ) $.
    jaoi.2 $e |- ( ch -> ps ) $.
    $( Inference disjoining the antecedents of two implications.  (Contributed
       by NM, 5-Apr-1994.) $)
    jaoi $p |- ( ( ph \/ ch ) -> ps ) $=
      ( wo wn pm2.53 syl6 pm2.61d2 ) ACFZABKAGCBACHEIDJ $.
  $}

  ${
    jaod.1 $e |- ( ph -> ( ps -> ch ) ) $.
    jaod.2 $e |- ( ph -> ( th -> ch ) ) $.
    $( Deduction disjoining the antecedents of two implications.  (Contributed
       by NM, 18-Aug-1994.) $)
    jaod $p |- ( ph -> ( ( ps \/ th ) -> ch ) ) $=
      ( wo wi com12 jaoi ) BDGACBACHDABCEIADCFIJI $.

    jaod.3 $e |- ( ph -> ( ps \/ th ) ) $.
    $( Eliminate a disjunction in a deduction.  (Contributed by Mario Carneiro,
       29-May-2016.) $)
    mpjaod $p |- ( ph -> ch ) $=
      ( wo jaod mpd ) ABDHCGABCDEFIJ $.
  $}

  $( Elimination of disjunction by denial of a disjunct.  Theorem *2.55 of
     [WhiteheadRussell] p. 107.  (Contributed by NM, 12-Aug-1994.)  (Proof
     shortened by Wolf Lammen, 21-Jul-2012.) $)
  orel1 $p |- ( -. ph -> ( ( ph \/ ps ) -> ps ) ) $=
    ( wo wn pm2.53 com12 ) ABCADBABEF $.

  $( Elimination of disjunction by denial of a disjunct.  Theorem *2.56 of
     [WhiteheadRussell] p. 107.  (Contributed by NM, 12-Aug-1994.)  (Proof
     shortened by Wolf Lammen, 5-Apr-2013.) $)
  orel2 $p |- ( -. ph -> ( ( ps \/ ph ) -> ps ) ) $=
    ( wn idd pm2.21 jaod ) ACZBBAGBDABEF $.

  $( Introduction of a disjunct.  Axiom *1.3 of [WhiteheadRussell] p. 96.
     (Contributed by NM, 30-Aug-1993.) $)
  olc $p |- ( ph -> ( ps \/ ph ) ) $=
    ( wn ax-1 orrd ) ABAABCDE $.

  $( Introduction of a disjunct.  Theorem *2.2 of [WhiteheadRussell] p. 104.
     (Contributed by NM, 30-Aug-1993.) $)
  orc $p |- ( ph -> ( ph \/ ps ) ) $=
    ( pm2.24 orrd ) AABABCD $.

  $( Axiom *1.4 of [WhiteheadRussell] p. 96.  (Contributed by NM,
     3-Jan-2005.) $)
  pm1.4 $p |- ( ( ph \/ ps ) -> ( ps \/ ph ) ) $=
    ( wo olc orc jaoi ) ABACBABDBAEF $.

  $( Commutative law for disjunction.  Theorem *4.31 of [WhiteheadRussell]
     p. 118.  (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Wolf
     Lammen, 15-Nov-2012.) $)
  orcom $p |- ( ( ph \/ ps ) <-> ( ps \/ ph ) ) $=
    ( wo pm1.4 impbii ) ABCBACABDBADE $.

  ${
    orcomd.1 $e |- ( ph -> ( ps \/ ch ) ) $.
    $( Commutation of disjuncts in consequent.  (Contributed by NM,
       2-Dec-2010.) $)
    orcomd $p |- ( ph -> ( ch \/ ps ) ) $=
      ( wo orcom sylib ) ABCECBEDBCFG $.
  $}

  ${
    orcoms.1 $e |- ( ( ph \/ ps ) -> ch ) $.
    $( Commutation of disjuncts in antecedent.  (Contributed by NM,
       2-Dec-2012.) $)
    orcoms $p |- ( ( ps \/ ph ) -> ch ) $=
      ( wo pm1.4 syl ) BAEABECBAFDG $.
  $}

  ${
    orci.1 $e |- ph $.
    $( Deduction introducing a disjunct.  (Contributed by NM, 19-Jan-2008.)
       (Proof shortened by Wolf Lammen, 14-Nov-2012.) $)
    orci $p |- ( ph \/ ps ) $=
      ( pm2.24i orri ) ABABCDE $.

    $( Deduction introducing a disjunct.  (Contributed by NM, 19-Jan-2008.)
       (Proof shortened by Wolf Lammen, 14-Nov-2012.) $)
    olci $p |- ( ps \/ ph ) $=
      ( wn a1i orri ) BAABDCEF $.
  $}

  ${
    orcd.1 $e |- ( ph -> ps ) $.
    $( Deduction introducing a disjunct.  A translation of natural deduction
       rule ` \/ ` IR ( ` \/ ` insertion right), see ~ natded .  (Contributed
       by NM, 20-Sep-2007.) $)
    orcd $p |- ( ph -> ( ps \/ ch ) ) $=
      ( wo orc syl ) ABBCEDBCFG $.

    $( Deduction introducing a disjunct.  A translation of natural deduction
       rule ` \/ ` IL ( ` \/ ` insertion left), see ~ natded .  (Contributed by
       NM, 11-Apr-2008.)  (Proof shortened by Wolf Lammen, 3-Oct-2013.) $)
    olcd $p |- ( ph -> ( ch \/ ps ) ) $=
      ( orcd orcomd ) ABCABCDEF $.
  $}

  ${
    orcs.1 $e |- ( ( ph \/ ps ) -> ch ) $.
    $( Deduction eliminating disjunct. _Notational convention_:  We sometimes
       suffix with "s" the label of an inference that manipulates an
       antecedent, leaving the consequent unchanged.  The "s" means that the
       inference eliminates the need for a syllogism ( ~ syl ) -type inference
       in a proof.  (Contributed by NM, 21-Jun-1994.) $)
    orcs $p |- ( ph -> ch ) $=
      ( wo orc syl ) AABECABFDG $.
  $}

  ${
    olcs.1 $e |- ( ( ph \/ ps ) -> ch ) $.
    $( Deduction eliminating disjunct.  (Contributed by NM, 21-Jun-1994.)
       (Proof shortened by Wolf Lammen, 3-Oct-2013.) $)
    olcs $p |- ( ps -> ch ) $=
      ( orcoms orcs ) BACABCDEF $.
  $}

  $( Theorem *2.07 of [WhiteheadRussell] p. 101.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.07 $p |- ( ph -> ( ph \/ ph ) ) $=
    ( olc ) AAB $.

  $( Theorem *2.45 of [WhiteheadRussell] p. 106.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.45 $p |- ( -. ( ph \/ ps ) -> -. ph ) $=
    ( wo orc con3i ) AABCABDE $.

  $( Theorem *2.46 of [WhiteheadRussell] p. 106.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.46 $p |- ( -. ( ph \/ ps ) -> -. ps ) $=
    ( wo olc con3i ) BABCBADE $.

  $( Theorem *2.47 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.47 $p |- ( -. ( ph \/ ps ) -> ( -. ph \/ ps ) ) $=
    ( wo wn pm2.45 orcd ) ABCDADBABEF $.

  $( Theorem *2.48 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.48 $p |- ( -. ( ph \/ ps ) -> ( ph \/ -. ps ) ) $=
    ( wo wn pm2.46 olcd ) ABCDBDAABEF $.

  $( Theorem *2.49 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.49 $p |- ( -. ( ph \/ ps ) -> ( -. ph \/ -. ps ) ) $=
    ( wo wn pm2.46 olcd ) ABCDBDADABEF $.

  $( Slight generalization of Theorem *2.67 of [WhiteheadRussell] p. 107.
     (Contributed by NM, 3-Jan-2005.) $)
  pm2.67-2 $p |- ( ( ( ph \/ ch ) -> ps ) -> ( ph -> ps ) ) $=
    ( wo orc imim1i ) AACDBACEF $.

  $( Theorem *2.67 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.67 $p |- ( ( ( ph \/ ps ) -> ps ) -> ( ph -> ps ) ) $=
    ( pm2.67-2 ) ABBC $.

  $( Theorem *2.25 of [WhiteheadRussell] p. 104.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.25 $p |- ( ph \/ ( ( ph \/ ps ) -> ps ) ) $=
    ( wo wi orel1 orri ) AABCBDABEF $.

  $( A wff is equivalent to its disjunction with falsehood.  Theorem *4.74 of
     [WhiteheadRussell] p. 121.  (Contributed by NM, 23-Mar-1995.)  (Proof
     shortened by Wolf Lammen, 18-Nov-2012.) $)
  biorf $p |- ( -. ph -> ( ps <-> ( ph \/ ps ) ) ) $=
    ( wn wo olc orel1 impbid2 ) ACBABDBAEABFG $.

  $( A wff is equivalent to its negated disjunction with falsehood.
     (Contributed by NM, 9-Jul-2012.) $)
  biortn $p |- ( ph -> ( ps <-> ( -. ph \/ ps ) ) ) $=
    ( wn wo wb notnot1 biorf syl ) AACZCBIBDEAFIBGH $.

  ${
    biorfi.1 $e |- -. ph $.
    $( A wff is equivalent to its disjunction with falsehood.  (Contributed by
       NM, 23-Mar-1995.) $)
    biorfi $p |- ( ps <-> ( ps \/ ph ) ) $=
      ( wn wo wb orc orel2 impbid2 ax-mp ) ADZBBAEZFCKBLBAGABHIJ $.
  $}

  $( Theorem *2.621 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.621 $p |- ( ( ph -> ps ) -> ( ( ph \/ ps ) -> ps ) ) $=
    ( wi id idd jaod ) ABCZABBGDGBEF $.

  $( Theorem *2.62 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 13-Dec-2013.) $)
  pm2.62 $p |- ( ( ph \/ ps ) -> ( ( ph -> ps ) -> ps ) ) $=
    ( wi wo pm2.621 com12 ) ABCABDBABEF $.

  $( Theorem *2.68 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.68 $p |- ( ( ( ph -> ps ) -> ps ) -> ( ph \/ ps ) ) $=
    ( wi jarl orrd ) ABCBCABABBDE $.

  $( Logical 'or' expressed in terms of implication only.  Theorem *5.25 of
     [WhiteheadRussell] p. 124.  (Contributed by NM, 12-Aug-2004.)  (Proof
     shortened by Wolf Lammen, 20-Oct-2012.) $)
  dfor2 $p |- ( ( ph \/ ps ) <-> ( ( ph -> ps ) -> ps ) ) $=
    ( wo wi pm2.62 pm2.68 impbii ) ABCABDBDABEABFG $.

  $( Implication in terms of disjunction.  Theorem *4.6 of [WhiteheadRussell]
     p. 120.  (Contributed by NM, 3-Jan-1993.) $)
  imor $p |- ( ( ph -> ps ) <-> ( -. ph \/ ps ) ) $=
    ( wi wn wo notnot imbi1i df-or bitr4i ) ABCADZDZBCJBEAKBAFGJBHI $.

  ${
    imori.1 $e |- ( ph -> ps ) $.
    $( Infer disjunction from implication.  (Contributed by NM,
       12-Mar-2012.) $)
    imori $p |- ( -. ph \/ ps ) $=
      ( wi wn wo imor mpbi ) ABDAEBFCABGH $.
  $}

  ${
    imorri.1 $e |- ( -. ph \/ ps ) $.
    $( Infer implication from disjunction.  (Contributed by Jonathan Ben-Naim,
       3-Jun-2011.) $)
    imorri $p |- ( ph -> ps ) $=
      ( wi wn wo imor mpbir ) ABDAEBFCABGH $.
  $}

  $( Law of excluded middle, also called the principle of _tertium non datur_.
     Theorem *2.11 of [WhiteheadRussell] p. 101.  It says that something is
     either true or not true; there are no in-between values of truth.  This is
     an essential distinction of our classical logic and is not a theorem of
     intuitionistic logic.  In intuitionistic logic, if this statement is true
     for some ` ph ` , then ` ph ` is decideable.  (Contributed by NM,
     29-Dec-1992.) $)
  exmid $p |- ( ph \/ -. ph ) $=
    ( wn id orri ) AABZECD $.

  $( Law of excluded middle in a context.  (Contributed by Mario Carneiro,
     9-Feb-2017.) $)
  exmidd $p |- ( ph -> ( ps \/ -. ps ) ) $=
    ( wn wo exmid a1i ) BBCDABEF $.

  $( Theorem *2.1 of [WhiteheadRussell] p. 101.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 23-Nov-2012.) $)
  pm2.1 $p |- ( -. ph \/ ph ) $=
    ( id imori ) AAABC $.

  $( Theorem *2.13 of [WhiteheadRussell] p. 101.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.13 $p |- ( ph \/ -. -. -. ph ) $=
    ( wn notnot1 orri ) AABZBBECD $.

  $( Theorem *4.62 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.62 $p |- ( ( ph -> -. ps ) <-> ( -. ph \/ -. ps ) ) $=
    ( wn imor ) ABCD $.

  $( Theorem *4.66 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.66 $p |- ( ( -. ph -> -. ps ) <-> ( ph \/ -. ps ) ) $=
    ( wn pm4.64 ) ABCD $.

  $( Theorem *4.63 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.63 $p |- ( -. ( ph -> -. ps ) <-> ( ph /\ ps ) ) $=
    ( wa wn wi df-an bicomi ) ABCABDEDABFG $.

  $( Express implication in terms of conjunction.  (Contributed by NM,
     9-Apr-1994.) $)
  imnan $p |- ( ( ph -> -. ps ) <-> -. ( ph /\ ps ) ) $=
    ( wa wn wi df-an con2bii ) ABCABDEABFG $.

  ${
    imnani.1 $e |- -. ( ph /\ ps ) $.
    $( Infer implication from negated conjunction.  (Contributed by Mario
       Carneiro, 28-Sep-2015.) $)
    imnani $p |- ( ph -> -. ps ) $=
      ( wn wi wa imnan mpbir ) ABDEABFDCABGH $.
  $}

  $( Express implication in terms of conjunction.  Theorem 3.4(27) of [Stoll]
     p. 176.  (Contributed by NM, 12-Mar-1993.)  (Proof shortened by Wolf
     Lammen, 30-Oct-2012.) $)
  iman $p |- ( ( ph -> ps ) <-> -. ( ph /\ -. ps ) ) $=
    ( wi wn wa notnot imbi2i imnan bitri ) ABCABDZDZCAJEDBKABFGAJHI $.

  $( Express conjunction in terms of implication.  (Contributed by NM,
     2-Aug-1994.) $)
  annim $p |- ( ( ph /\ -. ps ) <-> -. ( ph -> ps ) ) $=
    ( wi wn wa iman con2bii ) ABCABDEABFG $.

  $( Theorem *4.61 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.61 $p |- ( -. ( ph -> ps ) <-> ( ph /\ -. ps ) ) $=
    ( wn wa wi annim bicomi ) ABCDABECABFG $.

  $( Theorem *4.65 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.65 $p |- ( -. ( -. ph -> ps ) <-> ( -. ph /\ -. ps ) ) $=
    ( wn pm4.61 ) ACBD $.

  $( Theorem *4.67 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.67 $p |- ( -. ( -. ph -> -. ps ) <-> ( -. ph /\ ps ) ) $=
    ( wn pm4.63 ) ACBD $.

  ${
    imp.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Importation inference.  (Contributed by NM, 3-Jan-1993.)  (Proof
       shortened by Eric Schmidt, 22-Dec-2006.) $)
    imp $p |- ( ( ph /\ ps ) -> ch ) $=
      ( wa wn wi df-an impi sylbi ) ABEABFGFCABHABCDIJ $.

    $( Importation inference with commuted antecedents.  (Contributed by NM,
       25-May-2005.) $)
    impcom $p |- ( ( ps /\ ph ) -> ch ) $=
      ( com12 imp ) BACABCDEF $.
  $}

  ${
    imp3.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( Importation deduction.  (Contributed by NM, 31-Mar-1994.) $)
    impd $p |- ( ph -> ( ( ps /\ ch ) -> th ) ) $=
      ( wa wi com3l imp com12 ) BCFADBCADGABCDEHIJ $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp31 $p |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $=
      ( wa wi imp ) ABFCDABCDGEHH $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp32 $p |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $=
      ( wa impd imp ) ABCFDABCDEGH $.
  $}

  ${
    ex.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( Exportation inference.  (This theorem used to be labeled "exp" but was
       changed to "ex" so as not to conflict with the math token "exp", per the
       June 2006 Metamath spec change.)  A translation of natural deduction
       rule ` -> ` I ( ` -> ` introduction), see ~ natded .  (Contributed by
       NM, 3-Jan-1993.)  (Proof shortened by Eric Schmidt, 22-Dec-2006.) $)
    ex $p |- ( ph -> ( ps -> ch ) ) $=
      ( wn wi wa df-an sylbir expi ) ABCABEFEABGCABHDIJ $.

    $( Exportation inference with commuted antecedents.  (Contributed by NM,
       25-May-2005.) $)
    expcom $p |- ( ps -> ( ph -> ch ) ) $=
      ( ex com12 ) ABCABCDEF $.
  $}

  ${
    exp3a.1 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( Exportation deduction.  (Contributed by NM, 20-Aug-1993.) $)
    expd $p |- ( ph -> ( ps -> ( ch -> th ) ) ) $=
      ( wi wa com12 ex com3r ) BCADBCADFABCGDEHIJ $.

    $( A deduction version of exportation, followed by importation.
       (Contributed by NM, 6-Sep-2008.) $)
    expdimp $p |- ( ( ph /\ ps ) -> ( ch -> th ) ) $=
      ( wi expd imp ) ABCDFABCDEGH $.
  $}

  ${
    expcomd.1 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( Deduction form of ~ expcom .  (Contributed by Alan Sare,
       22-Jul-2012.) $)
    expcomd $p |- ( ph -> ( ch -> ( ps -> th ) ) ) $=
      ( expd com23 ) ABCDABCDEFG $.
  $}

  ${
    expdcom.1 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( Commuted form of ~ expd .  (Contributed by Alan Sare, 18-Mar-2012.) $)
    expdcom $p |- ( ps -> ( ch -> ( ph -> th ) ) ) $=
      ( expd com3l ) ABCDABCDEFG $.
  $}

  ${
    impancom.1 $e |- ( ( ph /\ ps ) -> ( ch -> th ) ) $.
    $( Mixed importation/commutation inference.  (Contributed by NM,
       22-Jun-2013.) $)
    impancom $p |- ( ( ph /\ ch ) -> ( ps -> th ) ) $=
      ( wi ex com23 imp ) ACBDFABCDABCDFEGHI $.
  $}

  ${
    con3dimp.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Variant of ~ con3d with importation.  (Contributed by Jonathan Ben-Naim,
       3-Jun-2011.) $)
    con3dimp $p |- ( ( ph /\ -. ch ) -> -. ps ) $=
      ( wn con3d imp ) ACEBEABCDFG $.
  $}

  ${
    pm2.01da.1 $e |- ( ( ph /\ ps ) -> -. ps ) $.
    $( Deduction based on reductio ad absurdum.  (Contributed by Mario
       Carneiro, 9-Feb-2017.) $)
    pm2.01da $p |- ( ph -> -. ps ) $=
      ( wn ex pm2.01d ) ABABBDCEF $.
  $}

  ${
    pm2.18da.1 $e |- ( ( ph /\ -. ps ) -> ps ) $.
    $( Deduction based on reductio ad absurdum.  (Contributed by Mario
       Carneiro, 9-Feb-2017.) $)
    pm2.18da $p |- ( ph -> ps ) $=
      ( wn ex pm2.18d ) ABABDBCEF $.
  $}

  $( Theorem *3.3 (Exp) of [WhiteheadRussell] p. 112.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 24-Mar-2013.) $)
  pm3.3 $p |- ( ( ( ph /\ ps ) -> ch ) -> ( ph -> ( ps -> ch ) ) ) $=
    ( wa wi id expd ) ABDCEZABCHFG $.

  $( Theorem *3.31 (Imp) of [WhiteheadRussell] p. 112.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 24-Mar-2013.) $)
  pm3.31 $p |- ( ( ph -> ( ps -> ch ) ) -> ( ( ph /\ ps ) -> ch ) ) $=
    ( wi id impd ) ABCDDZABCGEF $.

  $( Import-export theorem.  Part of Theorem *4.87 of [WhiteheadRussell]
     p. 122.  (Contributed by NM, 10-Jan-1993.)  (Proof shortened by Wolf
     Lammen, 24-Mar-2013.) $)
  impexp $p |- ( ( ( ph /\ ps ) -> ch ) <-> ( ph -> ( ps -> ch ) ) ) $=
    ( wa wi pm3.3 pm3.31 impbii ) ABDCEABCEEABCFABCGH $.

  $( Join antecedents with conjunction.  Theorem *3.2 of [WhiteheadRussell]
     p. 111.  (Contributed by NM, 5-Jan-1993.)  (Proof shortened by Wolf
     Lammen, 12-Nov-2012.) $)
  pm3.2 $p |- ( ph -> ( ps -> ( ph /\ ps ) ) ) $=
    ( wa id ex ) ABABCZFDE $.

  $( Join antecedents with conjunction.  Theorem *3.21 of [WhiteheadRussell]
     p. 111.  (Contributed by NM, 5-Aug-1993.) $)
  pm3.21 $p |- ( ph -> ( ps -> ( ps /\ ph ) ) ) $=
    ( wa pm3.2 com12 ) BABACBADE $.

  $( Theorem *3.22 of [WhiteheadRussell] p. 111.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 13-Nov-2012.) $)
  pm3.22 $p |- ( ( ph /\ ps ) -> ( ps /\ ph ) ) $=
    ( wa pm3.21 imp ) ABBACABDE $.

  $( Commutative law for conjunction.  Theorem *4.3 of [WhiteheadRussell]
     p. 118.  (Contributed by NM, 25-Jun-1998.)  (Proof shortened by Wolf
     Lammen, 4-Nov-2012.) $)
  ancom $p |- ( ( ph /\ ps ) <-> ( ps /\ ph ) ) $=
    ( wa pm3.22 impbii ) ABCBACABDBADE $.

  ${
    ancomd.1 $e |- ( ph -> ( ps /\ ch ) ) $.
    $( Commutation of conjuncts in consequent.  (Contributed by Jeff Hankins,
       14-Aug-2009.) $)
    ancomd $p |- ( ph -> ( ch /\ ps ) ) $=
      ( wa ancom sylib ) ABCECBEDBCFG $.
  $}

  $( Closed form of ~ ancoms .  (Contributed by Alan Sare, 31-Dec-2011.) $)
  ancomst $p |- ( ( ( ph /\ ps ) -> ch ) <-> ( ( ps /\ ph ) -> ch ) ) $=
    ( wa ancom imbi1i ) ABDBADCABEF $.

  ${
    ancoms.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( Inference commuting conjunction in antecedent.  (Contributed by NM,
       21-Apr-1994.) $)
    ancoms $p |- ( ( ps /\ ph ) -> ch ) $=
      ( expcom imp ) BACABCDEF $.
  $}

  ${
    ancomsd.1 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( Deduction commuting conjunction in antecedent.  (Contributed by NM,
       12-Dec-2004.) $)
    ancomsd $p |- ( ph -> ( ( ch /\ ps ) -> th ) ) $=
      ( wa ancom syl5bi ) CBFBCFADCBGEH $.
  $}

  ${
    pm3.2i.1 $e |- ph $.
    pm3.2i.2 $e |- ps $.
    $( Infer conjunction of premises.  (Contributed by NM, 21-Jun-1993.) $)
    pm3.2i $p |- ( ph /\ ps ) $=
      ( wa pm3.2 mp2 ) ABABECDABFG $.
  $}

  $( Nested conjunction of antecedents.  (Contributed by NM, 4-Jan-1993.) $)
  pm3.43i $p |- ( ( ph -> ps )
      -> ( ( ph -> ch ) -> ( ph -> ( ps /\ ch ) ) ) ) $=
    ( wa pm3.2 imim3i ) BCBCDABCEF $.

  $( Elimination of a conjunct.  Theorem *3.26 (Simp) of [WhiteheadRussell]
     p. 112.  (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Wolf
     Lammen, 13-Nov-2012.) $)
  simpl $p |- ( ( ph /\ ps ) -> ph ) $=
    ( ax-1 imp ) ABAABCD $.

  ${
    simpli.1 $e |- ( ph /\ ps ) $.
    $( Inference eliminating a conjunct.  (Contributed by NM, 15-Jun-1994.) $)
    simpli $p |- ph $=
      ( wa simpl ax-mp ) ABDACABEF $.
  $}

  ${
    simpld.1 $e |- ( ph -> ( ps /\ ch ) ) $.
    $( Deduction eliminating a conjunct.  A translation of natural deduction
       rule ` /\ ` EL ( ` /\ ` elimination left), see ~ natded .  (Contributed
       by NM, 26-May-1993.) $)
    simpld $p |- ( ph -> ps ) $=
      ( wa simpl syl ) ABCEBDBCFG $.
  $}

  ${
    simplbi.1 $e |- ( ph <-> ( ps /\ ch ) ) $.
    $( Deduction eliminating a conjunct.  (Contributed by NM, 27-May-1998.) $)
    simplbi $p |- ( ph -> ps ) $=
      ( wa biimpi simpld ) ABCABCEDFG $.
  $}

  $( Elimination of a conjunct.  Theorem *3.27 (Simp) of [WhiteheadRussell]
     p. 112.  (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Wolf
     Lammen, 13-Nov-2012.) $)
  simpr $p |- ( ( ph /\ ps ) -> ps ) $=
    ( idd imp ) ABBABCD $.

  ${
    simpri.1 $e |- ( ph /\ ps ) $.
    $( Inference eliminating a conjunct.  (Contributed by NM, 15-Jun-1994.) $)
    simpri $p |- ps $=
      ( wa simpr ax-mp ) ABDBCABEF $.
  $}

  ${
    simprd.1 $e |- ( ph -> ( ps /\ ch ) ) $.
    $( Deduction eliminating a conjunct.  (Contributed by NM, 14-May-1993.)  A
       translation of natural deduction rule ` /\ ` ER ( ` /\ ` elimination
       right), see ~ natded .  (Proof shortened by Wolf Lammen, 3-Oct-2013.) $)
    simprd $p |- ( ph -> ch ) $=
      ( ancomd simpld ) ACBABCDEF $.
  $}

  ${
    simprbi.1 $e |- ( ph <-> ( ps /\ ch ) ) $.
    $( Deduction eliminating a conjunct.  (Contributed by NM, 27-May-1998.) $)
    simprbi $p |- ( ph -> ch ) $=
      ( wa biimpi simprd ) ABCABCEDFG $.
  $}

  ${
    adantr.1 $e |- ( ph -> ps ) $.
    $( Inference adding a conjunct to the right of an antecedent.  (Contributed
       by NM, 30-Aug-1993.) $)
    adantr $p |- ( ( ph /\ ch ) -> ps ) $=
      ( a1d imp ) ACBABCDEF $.
  $}

  ${
    adantl.1 $e |- ( ph -> ps ) $.
    $( Inference adding a conjunct to the left of an antecedent.  (Contributed
       by NM, 30-Aug-1993.)  (Proof shortened by Wolf Lammen, 23-Nov-2012.) $)
    adantl $p |- ( ( ch /\ ph ) -> ps ) $=
      ( adantr ancoms ) ACBABCDEF $.
  $}

  ${
    adantld.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction adding a conjunct to the left of an antecedent.  (Contributed
       by NM, 4-May-1994.)  (Proof shortened by Wolf Lammen, 20-Dec-2012.) $)
    adantld $p |- ( ph -> ( ( th /\ ps ) -> ch ) ) $=
      ( wa simpr syl5 ) DBFBACDBGEH $.
  $}

  ${
    adantrd.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction adding a conjunct to the right of an antecedent.  (Contributed
       by NM, 4-May-1994.) $)
    adantrd $p |- ( ph -> ( ( ps /\ th ) -> ch ) ) $=
      ( wa simpl syl5 ) BDFBACBDGEH $.
  $}

  ${
    mpan9.1 $e |- ( ph -> ps ) $.
    mpan9.2 $e |- ( ch -> ( ps -> th ) ) $.
    $( Modus ponens conjoining dissimilar antecedents.  (Contributed by NM,
       1-Feb-2008.)  (Proof shortened by Andrew Salmon, 7-May-2011.) $)
    mpan9 $p |- ( ( ph /\ ch ) -> th ) $=
      ( syl5 impcom ) CADABCDEFGH $.
  $}

  ${
    syldan.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    syldan.2 $e |- ( ( ph /\ ch ) -> th ) $.
    $( A syllogism deduction with conjoined antecedents.  (Contributed by NM,
       24-Feb-2005.)  (Proof shortened by Wolf Lammen, 6-Apr-2013.) $)
    syldan $p |- ( ( ph /\ ps ) -> th ) $=
      ( wa expcom adantrd mpcom ) CABGDECADBACDFHIJ $.
  $}

  ${
    sylan.1 $e |- ( ph -> ps ) $.
    sylan.2 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A syllogism inference.  (Contributed by NM, 21-Apr-1994.)  (Proof
       shortened by Wolf Lammen, 22-Nov-2012.) $)
    sylan $p |- ( ( ph /\ ch ) -> th ) $=
      ( expcom mpan9 ) ABCDEBCDFGH $.
  $}

  ${
    sylanb.1 $e |- ( ph <-> ps ) $.
    sylanb.2 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A syllogism inference.  (Contributed by NM, 18-May-1994.) $)
    sylanb $p |- ( ( ph /\ ch ) -> th ) $=
      ( biimpi sylan ) ABCDABEGFH $.
  $}

  ${
    sylanbr.1 $e |- ( ps <-> ph ) $.
    sylanbr.2 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A syllogism inference.  (Contributed by NM, 18-May-1994.) $)
    sylanbr $p |- ( ( ph /\ ch ) -> th ) $=
      ( biimpri sylan ) ABCDBAEGFH $.
  $}

  ${
    sylan2.1 $e |- ( ph -> ch ) $.
    sylan2.2 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A syllogism inference.  (Contributed by NM, 21-Apr-1994.)  (Proof
       shortened by Wolf Lammen, 22-Nov-2012.) $)
    sylan2 $p |- ( ( ps /\ ph ) -> th ) $=
      ( adantl syldan ) BACDACBEGFH $.
  $}

  ${
    sylan2b.1 $e |- ( ph <-> ch ) $.
    sylan2b.2 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A syllogism inference.  (Contributed by NM, 21-Apr-1994.) $)
    sylan2b $p |- ( ( ps /\ ph ) -> th ) $=
      ( biimpi sylan2 ) ABCDACEGFH $.
  $}

  ${
    sylan2br.1 $e |- ( ch <-> ph ) $.
    sylan2br.2 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A syllogism inference.  (Contributed by NM, 21-Apr-1994.) $)
    sylan2br $p |- ( ( ps /\ ph ) -> th ) $=
      ( biimpri sylan2 ) ABCDCAEGFH $.
  $}

  ${
    syl2an.1 $e |- ( ph -> ps ) $.
    syl2an.2 $e |- ( ta -> ch ) $.
    syl2an.3 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A double syllogism inference.  (Contributed by NM, 31-Jan-1997.) $)
    syl2an $p |- ( ( ph /\ ta ) -> th ) $=
      ( sylan sylan2 ) EACDGABCDFHIJ $.

    $( A double syllogism inference.  (Contributed by NM, 17-Sep-2013.) $)
    syl2anr $p |- ( ( ta /\ ph ) -> th ) $=
      ( syl2an ancoms ) AEDABCDEFGHIJ $.
  $}

  ${
    syl2anb.1 $e |- ( ph <-> ps ) $.
    syl2anb.2 $e |- ( ta <-> ch ) $.
    syl2anb.3 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A double syllogism inference.  (Contributed by NM, 29-Jul-1999.) $)
    syl2anb $p |- ( ( ph /\ ta ) -> th ) $=
      ( sylanb sylan2b ) EACDGABCDFHIJ $.
  $}

  ${
    syl2anbr.1 $e |- ( ps <-> ph ) $.
    syl2anbr.2 $e |- ( ch <-> ta ) $.
    syl2anbr.3 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A double syllogism inference.  (Contributed by NM, 29-Jul-1999.) $)
    syl2anbr $p |- ( ( ph /\ ta ) -> th ) $=
      ( sylanbr sylan2br ) EACDGABCDFHIJ $.
  $}

  ${
    syland.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syland.2 $e |- ( ph -> ( ( ch /\ th ) -> ta ) ) $.
    $( A syllogism deduction.  (Contributed by NM, 15-Dec-2004.) $)
    syland $p |- ( ph -> ( ( ps /\ th ) -> ta ) ) $=
      ( wi expd syld impd ) ABDEABCDEHFACDEGIJK $.
  $}

  ${
    sylan2d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    sylan2d.2 $e |- ( ph -> ( ( th /\ ch ) -> ta ) ) $.
    $( A syllogism deduction.  (Contributed by NM, 15-Dec-2004.) $)
    sylan2d $p |- ( ph -> ( ( th /\ ps ) -> ta ) ) $=
      ( ancomsd syland ) ABDEABCDEFADCEGHIH $.
  $}

  ${
    syl2and.1 $e |- ( ph -> ( ps -> ch ) ) $.
    syl2and.2 $e |- ( ph -> ( th -> ta ) ) $.
    syl2and.3 $e |- ( ph -> ( ( ch /\ ta ) -> et ) ) $.
    $( A syllogism deduction.  (Contributed by NM, 15-Dec-2004.) $)
    syl2and $p |- ( ph -> ( ( ps /\ th ) -> et ) ) $=
      ( sylan2d syland ) ABCDFGADECFHIJK $.
  $}

  ${
    biimpa.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Inference from a logical equivalence.  (Contributed by NM,
       3-May-1994.) $)
    biimpa $p |- ( ( ph /\ ps ) -> ch ) $=
      ( biimpd imp ) ABCABCDEF $.

    $( Inference from a logical equivalence.  (Contributed by NM,
       3-May-1994.) $)
    biimpar $p |- ( ( ph /\ ch ) -> ps ) $=
      ( biimprd imp ) ACBABCDEF $.

    $( Inference from a logical equivalence.  (Contributed by NM,
       3-May-1994.) $)
    biimpac $p |- ( ( ps /\ ph ) -> ch ) $=
      ( biimpcd imp ) BACABCDEF $.

    $( Inference from a logical equivalence.  (Contributed by NM,
       3-May-1994.) $)
    biimparc $p |- ( ( ch /\ ph ) -> ps ) $=
      ( biimprcd imp ) CABABCDEF $.
  $}

  $( Negated conjunction in terms of disjunction (De Morgan's law).  Theorem
     *4.51 of [WhiteheadRussell] p. 120.  (Contributed by NM, 14-May-1993.)
     (Proof shortened by Andrew Salmon, 13-May-2011.) $)
  ianor $p |- ( -. ( ph /\ ps ) <-> ( -. ph \/ -. ps ) ) $=
    ( wa wn wi wo imnan pm4.62 bitr3i ) ABCDABDZEADJFABGABHI $.

  $( Conjunction in terms of disjunction (De Morgan's law).  Theorem *4.5 of
     [WhiteheadRussell] p. 120.  (Contributed by NM, 3-Jan-1993.)  (Proof
     shortened by Wolf Lammen, 3-Nov-2012.) $)
  anor $p |- ( ( ph /\ ps ) <-> -. ( -. ph \/ -. ps ) ) $=
    ( wn wo wa ianor bicomi con2bii ) ACBCDZABEZJCIABFGH $.

  $( Negated disjunction in terms of conjunction (De Morgan's law).  Compare
     Theorem *4.56 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-1993.)  (Proof shortened by Andrew Salmon, 7-May-2011.) $)
  ioran $p |- ( -. ( ph \/ ps ) <-> ( -. ph /\ -. ps ) ) $=
    ( wn wi wa wo pm4.65 pm4.64 xchnxbi ) ACZBDJBCEABFABGABHI $.

  $( Theorem *4.52 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 5-Nov-2012.) $)
  pm4.52 $p |- ( ( ph /\ -. ps ) <-> -. ( -. ph \/ ps ) ) $=
    ( wn wa wi wo annim imor xchbinx ) ABCDABEACBFABGABHI $.

  $( Theorem *4.53 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.53 $p |- ( -. ( ph /\ -. ps ) <-> ( -. ph \/ ps ) ) $=
    ( wn wo wa pm4.52 con2bii bicomi ) ACBDZABCEZCJIABFGH $.

  $( Theorem *4.54 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 5-Nov-2012.) $)
  pm4.54 $p |- ( ( -. ph /\ ps ) <-> -. ( ph \/ -. ps ) ) $=
    ( wn wa wi wo df-an pm4.66 xchbinx ) ACZBDJBCZEAKFJBGABHI $.

  $( Theorem *4.55 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.55 $p |- ( -. ( -. ph /\ ps ) <-> ( ph \/ -. ps ) ) $=
    ( wn wo wa pm4.54 con2bii bicomi ) ABCDZACBEZCJIABFGH $.

  $( Theorem *4.56 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.56 $p |- ( ( -. ph /\ -. ps ) <-> -. ( ph \/ ps ) ) $=
    ( wo wn wa ioran bicomi ) ABCDADBDEABFG $.

  $( Disjunction in terms of conjunction (De Morgan's law).  Compare Theorem
     *4.57 of [WhiteheadRussell] p. 120.  (Contributed by NM, 3-Jan-1993.)
     (Proof shortened by Andrew Salmon, 7-May-2011.) $)
  oran $p |- ( ( ph \/ ps ) <-> -. ( -. ph /\ -. ps ) ) $=
    ( wn wa wo pm4.56 con2bii ) ACBCDABEABFG $.

  $( Theorem *4.57 of [WhiteheadRussell] p. 120.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.57 $p |- ( -. ( -. ph /\ -. ps ) <-> ( ph \/ ps ) ) $=
    ( wo wn wa oran bicomi ) ABCADBDEDABFG $.

  $( Theorem *3.1 of [WhiteheadRussell] p. 111.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.1 $p |- ( ( ph /\ ps ) -> -. ( -. ph \/ -. ps ) ) $=
    ( wa wn wo anor biimpi ) ABCADBDEDABFG $.

  $( Theorem *3.11 of [WhiteheadRussell] p. 111.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.11 $p |- ( -. ( -. ph \/ -. ps ) -> ( ph /\ ps ) ) $=
    ( wa wn wo anor biimpri ) ABCADBDEDABFG $.

  $( Theorem *3.12 of [WhiteheadRussell] p. 111.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.12 $p |- ( ( -. ph \/ -. ps ) \/ ( ph /\ ps ) ) $=
    ( wn wo wa pm3.11 orri ) ACBCDABEABFG $.

  $( Theorem *3.13 of [WhiteheadRussell] p. 111.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.13 $p |- ( -. ( ph /\ ps ) -> ( -. ph \/ -. ps ) ) $=
    ( wn wo wa pm3.11 con1i ) ACBCDABEABFG $.

  $( Theorem *3.14 of [WhiteheadRussell] p. 111.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.14 $p |- ( ( -. ph \/ -. ps ) -> -. ( ph /\ ps ) ) $=
    ( wa wn wo pm3.1 con2i ) ABCADBDEABFG $.

  $( Introduction of antecedent as conjunct.  Theorem *4.73 of
     [WhiteheadRussell] p. 121.  (Contributed by NM, 30-Mar-1994.) $)
  iba $p |- ( ph -> ( ps <-> ( ps /\ ph ) ) ) $=
    ( wa pm3.21 simpl impbid1 ) ABBACABDBAEF $.

  $( Introduction of antecedent as conjunct.  (Contributed by NM,
     5-Dec-1995.) $)
  ibar $p |- ( ph -> ( ps <-> ( ph /\ ps ) ) ) $=
    ( wa pm3.2 simpr impbid1 ) ABABCABDABEF $.

  ${
    biantru.1 $e |- ph $.
    $( A wff is equivalent to its conjunction with truth.  (Contributed by NM,
       26-May-1993.) $)
    biantru $p |- ( ps <-> ( ps /\ ph ) ) $=
      ( wa wb iba ax-mp ) ABBADECABFG $.
  $}

  ${
    biantrur.1 $e |- ph $.
    $( A wff is equivalent to its conjunction with truth.  (Contributed by NM,
       3-Aug-1994.) $)
    biantrur $p |- ( ps <-> ( ph /\ ps ) ) $=
      ( wa wb ibar ax-mp ) ABABDECABFG $.
  $}

  ${
    biantrud.1 $e |- ( ph -> ps ) $.
    $( A wff is equivalent to its conjunction with truth.  (Contributed by NM,
       2-Aug-1994.)  (Proof shortened by Wolf Lammen, 23-Oct-2013.) $)
    biantrud $p |- ( ph -> ( ch <-> ( ch /\ ps ) ) ) $=
      ( wa wb iba syl ) ABCCBEFDBCGH $.

    $( A wff is equivalent to its conjunction with truth.  (Contributed by NM,
       1-May-1995.)  (Proof shortened by Andrew Salmon, 7-May-2011.) $)
    biantrurd $p |- ( ph -> ( ch <-> ( ps /\ ch ) ) ) $=
      ( wa wb ibar syl ) ABCBCEFDBCGH $.
  $}

  ${
    jaao.1 $e |- ( ph -> ( ps -> ch ) ) $.
    jaao.2 $e |- ( th -> ( ta -> ch ) ) $.
    $( Inference conjoining and disjoining the antecedents of two
       implications.  (Contributed by NM, 30-Sep-1999.) $)
    jaao $p |- ( ( ph /\ th ) -> ( ( ps \/ ta ) -> ch ) ) $=
      ( wa wi adantr adantl jaod ) ADHBCEABCIDFJDECIAGKL $.

    $( Inference disjoining and conjoining the antecedents of two
       implications.  (Contributed by Stefan Allan, 1-Nov-2008.) $)
    jaoa $p |- ( ( ph \/ th ) -> ( ( ps /\ ta ) -> ch ) ) $=
      ( wa wi adantrd adantld jaoi ) ABEHCIDABCEFJDECBGKL $.
  $}

  $( Theorem *3.44 of [WhiteheadRussell] p. 113.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 3-Oct-2013.) $)
  pm3.44 $p |- ( ( ( ps -> ph ) /\ ( ch -> ph ) )
      -> ( ( ps \/ ch ) -> ph ) ) $=
    ( wi id jaao ) BADZBACADZCGEHEF $.

  $( Disjunction of antecedents.  Compare Theorem *3.44 of [WhiteheadRussell]
     p. 113.  (Contributed by NM, 5-Apr-1994.)  (Proof shortened by Wolf
     Lammen, 4-Apr-2013.) $)
  jao $p |- ( ( ph -> ps ) -> ( ( ch -> ps ) -> ( ( ph \/ ch ) -> ps ) ) ) $=
    ( wi wo pm3.44 ex ) ABDCBDACEBDBACFG $.

  $( Axiom *1.2 of [WhiteheadRussell] p. 96, which they call "Taut".
     (Contributed by NM, 3-Jan-2005.) $)
  pm1.2 $p |- ( ( ph \/ ph ) -> ph ) $=
    ( id jaoi ) AAAABZDC $.

  $( Idempotent law for disjunction.  Theorem *4.25 of [WhiteheadRussell]
     p. 117.  (Contributed by NM, 11-May-1993.)  (Proof shortened by Andrew
     Salmon, 16-Apr-2011.)  (Proof shortened by Wolf Lammen, 10-Mar-2013.) $)
  oridm $p |- ( ( ph \/ ph ) <-> ph ) $=
    ( wo pm1.2 pm2.07 impbii ) AABAACADE $.

  $( Theorem *4.25 of [WhiteheadRussell] p. 117.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.25 $p |- ( ph <-> ( ph \/ ph ) ) $=
    ( wo oridm bicomi ) AABAACD $.

  ${
    orim12i.1 $e |- ( ph -> ps ) $.
    orim12i.2 $e |- ( ch -> th ) $.
    $( Disjoin antecedents and consequents of two premises.  (Contributed by
       NM, 6-Jun-1994.)  (Proof shortened by Wolf Lammen, 25-Jul-2012.) $)
    orim12i $p |- ( ( ph \/ ch ) -> ( ps \/ th ) ) $=
      ( wo orcd olcd jaoi ) ABDGCABDEHCDBFIJ $.
  $}

  ${
    orim1i.1 $e |- ( ph -> ps ) $.
    $( Introduce disjunct to both sides of an implication.  (Contributed by NM,
       6-Jun-1994.) $)
    orim1i $p |- ( ( ph \/ ch ) -> ( ps \/ ch ) ) $=
      ( id orim12i ) ABCCDCEF $.

    $( Introduce disjunct to both sides of an implication.  (Contributed by NM,
       6-Jun-1994.) $)
    orim2i $p |- ( ( ch \/ ph ) -> ( ch \/ ps ) ) $=
      ( id orim12i ) CCABCEDF $.
  $}

  ${
    orbi2i.1 $e |- ( ph <-> ps ) $.
    $( Inference adding a left disjunct to both sides of a logical
       equivalence.  (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Wolf
       Lammen, 12-Dec-2012.) $)
    orbi2i $p |- ( ( ch \/ ph ) <-> ( ch \/ ps ) ) $=
      ( wo biimpi orim2i biimpri impbii ) CAECBEABCABDFGBACABDHGI $.

    $( Inference adding a right disjunct to both sides of a logical
       equivalence.  (Contributed by NM, 3-Jan-1993.) $)
    orbi1i $p |- ( ( ph \/ ch ) <-> ( ps \/ ch ) ) $=
      ( wo orcom orbi2i 3bitri ) ACECAECBEBCEACFABCDGCBFH $.
  $}

  ${
    orbi12i.1 $e |- ( ph <-> ps ) $.
    orbi12i.2 $e |- ( ch <-> th ) $.
    $( Infer the disjunction of two equivalences.  (Contributed by NM,
       3-Jan-1993.) $)
    orbi12i $p |- ( ( ph \/ ch ) <-> ( ps \/ th ) ) $=
      ( wo orbi2i orbi1i bitri ) ACGADGBDGCDAFHABDEIJ $.
  $}

  $( Axiom *1.5 (Assoc) of [WhiteheadRussell] p. 96.  (Contributed by NM,
     3-Jan-2005.) $)
  pm1.5 $p |- ( ( ph \/ ( ps \/ ch ) ) -> ( ps \/ ( ph \/ ch ) ) ) $=
    ( wo orc olcd olc orim2i jaoi ) ABACDZDBCDAJBACEFCJBCAGHI $.

  $( Swap two disjuncts.  (Contributed by NM, 5-Aug-1993.)  (Proof shortened by
     Wolf Lammen, 14-Nov-2012.) $)
  or12 $p |- ( ( ph \/ ( ps \/ ch ) ) <-> ( ps \/ ( ph \/ ch ) ) ) $=
    ( wo pm1.5 impbii ) ABCDDBACDDABCEBACEF $.

  $( Associative law for disjunction.  Theorem *4.33 of [WhiteheadRussell]
     p. 118.  (Contributed by NM, 5-Aug-1993.)  (Proof shortened by Andrew
     Salmon, 26-Jun-2011.) $)
  orass $p |- ( ( ( ph \/ ps ) \/ ch ) <-> ( ph \/ ( ps \/ ch ) ) ) $=
    ( wo orcom or12 orbi2i 3bitri ) ABDZCDCIDACBDZDABCDZDICECABFJKACBEGH $.

  $( Theorem *2.31 of [WhiteheadRussell] p. 104.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.31 $p |- ( ( ph \/ ( ps \/ ch ) ) -> ( ( ph \/ ps ) \/ ch ) ) $=
    ( wo orass biimpri ) ABDCDABCDDABCEF $.

  $( Theorem *2.32 of [WhiteheadRussell] p. 105.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.32 $p |- ( ( ( ph \/ ps ) \/ ch ) -> ( ph \/ ( ps \/ ch ) ) ) $=
    ( wo orass biimpi ) ABDCDABCDDABCEF $.

  $( A rearrangement of disjuncts.  (Contributed by NM, 18-Oct-1995.)  (Proof
     shortened by Andrew Salmon, 26-Jun-2011.) $)
  or32 $p |- ( ( ( ph \/ ps ) \/ ch ) <-> ( ( ph \/ ch ) \/ ps ) ) $=
    ( wo orass or12 orcom 3bitri ) ABDCDABCDDBACDZDIBDABCEABCFBIGH $.

  $( Rearrangement of 4 disjuncts.  (Contributed by NM, 12-Aug-1994.) $)
  or4 $p |- ( ( ( ph \/ ps ) \/ ( ch \/ th ) ) <->
                ( ( ph \/ ch ) \/ ( ps \/ th ) ) ) $=
    ( wo or12 orbi2i orass 3bitr4i ) ABCDEZEZEACBDEZEZEABEJEACELEKMABCDFGABJHAC
    LHI $.

  $( Rearrangement of 4 disjuncts.  (Contributed by NM, 10-Jan-2005.) $)
  or42 $p |- ( ( ( ph \/ ps ) \/ ( ch \/ th ) ) <->
                 ( ( ph \/ ch ) \/ ( th \/ ps ) ) ) $=
    ( wo or4 orcom orbi2i bitri ) ABECDEEACEZBDEZEJDBEZEABCDFKLJBDGHI $.

  $( Distribution of disjunction over disjunction.  (Contributed by NM,
     25-Feb-1995.) $)
  orordi $p |- ( ( ph \/ ( ps \/ ch ) ) <->
               ( ( ph \/ ps ) \/ ( ph \/ ch ) ) ) $=
    ( wo oridm orbi1i or4 bitr3i ) ABCDZDAADZIDABDACDDJAIAEFAABCGH $.

  $( Distribution of disjunction over disjunction.  (Contributed by NM,
     25-Feb-1995.) $)
  orordir $p |- ( ( ( ph \/ ps ) \/ ch ) <->
               ( ( ph \/ ch ) \/ ( ps \/ ch ) ) ) $=
    ( wo oridm orbi2i or4 bitr3i ) ABDZCDICCDZDACDBCDDJCICEFABCCGH $.

  ${
    jca.1 $e |- ( ph -> ps ) $.
    jca.2 $e |- ( ph -> ch ) $.
    $( Deduce conjunction of the consequents of two implications ("join
       consequents with 'and'").  Equivalent to the natural deduction rule
       ` /\ ` I ( ` /\ ` introduction), see ~ natded .  (Contributed by NM,
       3-Jan-1993.)  (Proof shortened by Wolf Lammen, 25-Oct-2012.) $)
    jca $p |- ( ph -> ( ps /\ ch ) ) $=
      ( wa pm3.2 sylc ) ABCBCFDEBCGH $.
  $}

  ${
    jcad.1 $e |- ( ph -> ( ps -> ch ) ) $.
    jcad.2 $e |- ( ph -> ( ps -> th ) ) $.
    $( Deduction conjoining the consequents of two implications.  (Contributed
       by NM, 15-Jul-1993.)  (Proof shortened by Wolf Lammen, 23-Jul-2013.) $)
    jcad $p |- ( ph -> ( ps -> ( ch /\ th ) ) ) $=
      ( wa pm3.2 syl6c ) ABCDCDGEFCDHI $.
  $}

  ${
    jca31.1 $e |- ( ph -> ps ) $.
    jca31.2 $e |- ( ph -> ch ) $.
    jca31.3 $e |- ( ph -> th ) $.
    $( Join three consequents.  (Contributed by Jeff Hankins, 1-Aug-2009.) $)
    jca31 $p |- ( ph -> ( ( ps /\ ch ) /\ th ) ) $=
      ( wa jca ) ABCHDABCEFIGI $.

    $( Join three consequents.  (Contributed by FL, 1-Aug-2009.) $)
    jca32 $p |- ( ph -> ( ps /\ ( ch /\ th ) ) ) $=
      ( wa jca ) ABCDHEACDFGII $.
  $}

  ${
    jcai.1 $e |- ( ph -> ps ) $.
    jcai.2 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction replacing implication with conjunction.  (Contributed by NM,
       15-Jul-1993.) $)
    jcai $p |- ( ph -> ( ps /\ ch ) ) $=
      ( mpd jca ) ABCDABCDEFG $.
  $}

  ${
    jctil.1 $e |- ( ph -> ps ) $.
    jctil.2 $e |- ch $.
    $( Inference conjoining a theorem to left of consequent in an implication.
       (Contributed by NM, 31-Dec-1993.) $)
    jctil $p |- ( ph -> ( ch /\ ps ) ) $=
      ( a1i jca ) ACBCAEFDG $.

    $( Inference conjoining a theorem to right of consequent in an
       implication.  (Contributed by NM, 31-Dec-1993.) $)
    jctir $p |- ( ph -> ( ps /\ ch ) ) $=
      ( a1i jca ) ABCDCAEFG $.
  $}

  ${
    jccir.1 $e |- ( ph -> ps ) $.
    jccir.2 $e |- ( ps -> ch ) $.
    $( Inference conjoining a consequent of a consequent to the right of the
       consequent in an implication.  See also ~ ex-natded5.3i .  (Contributed
       by Mario Carneiro, 9-Feb-2017.)  (Revised by AV, 20-Aug-2019.) $)
    jccir $p |- ( ph -> ( ps /\ ch ) ) $=
      ( syl jca ) ABCDABCDEFG $.

    $( Inference conjoining a consequent of a consequent to the left of the
       consequent in an implication.  Remark:  One can also prove this theorem
       using ~ syl and ~ jca (as done in ~ jccir ), which would be 4 bytes
       shorter, but one step longer than the current proof.
       (Proof modification is discouraged.)  (Contributed by AV,
       20-Aug-2019.) $)
    jccil $p |- ( ph -> ( ch /\ ps ) ) $=
      ( jccir ancomd ) ABCABCDEFG $.
  $}

  ${
    jctl.1 $e |- ps $.
    $( Inference conjoining a theorem to the left of a consequent.
       (Contributed by NM, 31-Dec-1993.)  (Proof shortened by Wolf Lammen,
       24-Oct-2012.) $)
    jctl $p |- ( ph -> ( ps /\ ph ) ) $=
      ( id jctil ) AABADCE $.

    $( Inference conjoining a theorem to the right of a consequent.
       (Contributed by NM, 18-Aug-1993.)  (Proof shortened by Wolf Lammen,
       24-Oct-2012.) $)
    jctr $p |- ( ph -> ( ph /\ ps ) ) $=
      ( id jctir ) AABADCE $.
  $}

  ${
    jctild.1 $e |- ( ph -> ( ps -> ch ) ) $.
    jctild.2 $e |- ( ph -> th ) $.
    $( Deduction conjoining a theorem to left of consequent in an implication.
       (Contributed by NM, 21-Apr-2005.) $)
    jctild $p |- ( ph -> ( ps -> ( th /\ ch ) ) ) $=
      ( a1d jcad ) ABDCADBFGEH $.
  $}

  ${
    jctird.1 $e |- ( ph -> ( ps -> ch ) ) $.
    jctird.2 $e |- ( ph -> th ) $.
    $( Deduction conjoining a theorem to right of consequent in an
       implication.  (Contributed by NM, 21-Apr-2005.) $)
    jctird $p |- ( ph -> ( ps -> ( ch /\ th ) ) ) $=
      ( a1d jcad ) ABCDEADBFGH $.
  $}

  ${
    syl6an.1 $e |- ( ph -> ps ) $.
    syl6an.2 $e |- ( ph -> ( ch -> th ) ) $.
    syl6an.3 $e |- ( ( ps /\ th ) -> ta ) $.
    $( A syllogism deduction combined with conjoining antecedents.
       (Contributed by Alan Sare, 28-Oct-2011.) $)
    syl6an $p |- ( ph -> ( ch -> ta ) ) $=
      ( wa jctild syl6 ) ACBDIEACDBGFJHK $.
  $}

  $( Conjoin antecedent to left of consequent.  (Contributed by NM,
     15-Aug-1994.) $)
  ancl $p |- ( ( ph -> ps ) -> ( ph -> ( ph /\ ps ) ) ) $=
    ( wa pm3.2 a2i ) ABABCABDE $.

  $( Conjoin antecedent to left of consequent.  Theorem *4.7 of
     [WhiteheadRussell] p. 120.  (Contributed by NM, 25-Jul-1999.)  (Proof
     shortened by Wolf Lammen, 24-Mar-2013.) $)
  anclb $p |- ( ( ph -> ps ) <-> ( ph -> ( ph /\ ps ) ) ) $=
    ( wa ibar pm5.74i ) ABABCABDE $.

  $( Theorem *5.42 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.42 $p |- ( ( ph -> ( ps -> ch ) ) <->
                ( ph -> ( ps -> ( ph /\ ch ) ) ) ) $=
    ( wi wa ibar imbi2d pm5.74i ) ABCDBACEZDACIBACFGH $.

  $( Conjoin antecedent to right of consequent.  (Contributed by NM,
     15-Aug-1994.) $)
  ancr $p |- ( ( ph -> ps ) -> ( ph -> ( ps /\ ph ) ) ) $=
    ( wa pm3.21 a2i ) ABBACABDE $.

  $( Conjoin antecedent to right of consequent.  (Contributed by NM,
     25-Jul-1999.)  (Proof shortened by Wolf Lammen, 24-Mar-2013.) $)
  ancrb $p |- ( ( ph -> ps ) <-> ( ph -> ( ps /\ ph ) ) ) $=
    ( wa iba pm5.74i ) ABBACABDE $.

  ${
    ancli.1 $e |- ( ph -> ps ) $.
    $( Deduction conjoining antecedent to left of consequent.  (Contributed by
       NM, 12-Aug-1993.) $)
    ancli $p |- ( ph -> ( ph /\ ps ) ) $=
      ( id jca ) AABADCE $.
  $}

  ${
    ancri.1 $e |- ( ph -> ps ) $.
    $( Deduction conjoining antecedent to right of consequent.  (Contributed by
       NM, 15-Aug-1994.) $)
    ancri $p |- ( ph -> ( ps /\ ph ) ) $=
      ( id jca ) ABACADE $.
  $}

  ${
    ancld.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction conjoining antecedent to left of consequent in nested
       implication.  (Contributed by NM, 15-Aug-1994.)  (Proof shortened by
       Wolf Lammen, 1-Nov-2012.) $)
    ancld $p |- ( ph -> ( ps -> ( ps /\ ch ) ) ) $=
      ( idd jcad ) ABBCABEDF $.
  $}

  ${
    ancrd.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction conjoining antecedent to right of consequent in nested
       implication.  (Contributed by NM, 15-Aug-1994.)  (Proof shortened by
       Wolf Lammen, 1-Nov-2012.) $)
    ancrd $p |- ( ph -> ( ps -> ( ch /\ ps ) ) ) $=
      ( idd jcad ) ABCBDABEF $.
  $}

  $( Conjoin antecedent to left of consequent in nested implication.
     (Contributed by NM, 10-Aug-1994.)  (Proof shortened by Wolf Lammen,
     14-Jul-2013.) $)
  anc2l $p |- ( ( ph -> ( ps -> ch ) ) -> ( ph -> ( ps -> ( ph /\ ch ) ) ) ) $=
    ( wi wa pm5.42 biimpi ) ABCDDABACEDDABCFG $.

  $( Conjoin antecedent to right of consequent in nested implication.
     (Contributed by NM, 15-Aug-1994.) $)
  anc2r $p |- ( ( ph -> ( ps -> ch ) ) -> ( ph -> ( ps -> ( ch /\ ph ) ) ) ) $=
    ( wi wa pm3.21 imim2d a2i ) ABCDBCAEZDACIBACFGH $.

  ${
    anc2li.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction conjoining antecedent to left of consequent in nested
       implication.  (Contributed by NM, 10-Aug-1994.)  (Proof shortened by
       Wolf Lammen, 7-Dec-2012.) $)
    anc2li $p |- ( ph -> ( ps -> ( ph /\ ch ) ) ) $=
      ( id jctild ) ABCADAEF $.
  $}

  ${
    anc2ri.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction conjoining antecedent to right of consequent in nested
       implication.  (Contributed by NM, 15-Aug-1994.)  (Proof shortened by
       Wolf Lammen, 7-Dec-2012.) $)
    anc2ri $p |- ( ph -> ( ps -> ( ch /\ ph ) ) ) $=
      ( id jctird ) ABCADAEF $.
  $}

  $( Theorem *3.41 of [WhiteheadRussell] p. 113.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.41 $p |- ( ( ph -> ch ) -> ( ( ph /\ ps ) -> ch ) ) $=
    ( wa simpl imim1i ) ABDACABEF $.

  $( Theorem *3.42 of [WhiteheadRussell] p. 113.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.42 $p |- ( ( ps -> ch ) -> ( ( ph /\ ps ) -> ch ) ) $=
    ( wa simpr imim1i ) ABDBCABEF $.

  $( Conjunction implies implication.  Theorem *3.4 of [WhiteheadRussell]
     p. 113.  (Contributed by NM, 31-Jul-1995.) $)
  pm3.4 $p |- ( ( ph /\ ps ) -> ( ph -> ps ) ) $=
    ( wa simpr a1d ) ABCBAABDE $.

  $( Conjunction with implication.  Compare Theorem *4.45 of [WhiteheadRussell]
     p. 119.  (Contributed by NM, 17-May-1998.) $)
  pm4.45im $p |- ( ph <-> ( ph /\ ( ps -> ph ) ) ) $=
    ( wi wa ax-1 ancli simpl impbii ) AABACZDAIABEFAIGH $.

  ${
    anim12d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    anim12d.2 $e |- ( ph -> ( th -> ta ) ) $.
    $( Conjoin antecedents and consequents in a deduction.  (Contributed by NM,
       3-Apr-1994.)  (Proof shortened by Wolf Lammen, 18-Dec-2013.) $)
    anim12d $p |- ( ph -> ( ( ps /\ th ) -> ( ch /\ ta ) ) ) $=
      ( wa idd syl2and ) ABCDECEHZFGAKIJ $.
  $}

  ${
    anim1d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Add a conjunct to right of antecedent and consequent in a deduction.
       (Contributed by NM, 3-Apr-1994.) $)
    anim1d $p |- ( ph -> ( ( ps /\ th ) -> ( ch /\ th ) ) ) $=
      ( idd anim12d ) ABCDDEADFG $.

    $( Add a conjunct to left of antecedent and consequent in a deduction.
       (Contributed by NM, 14-May-1993.) $)
    anim2d $p |- ( ph -> ( ( th /\ ps ) -> ( th /\ ch ) ) ) $=
      ( idd anim12d ) ADDBCADFEG $.
  $}

  ${
    anim12i.1 $e |- ( ph -> ps ) $.
    anim12i.2 $e |- ( ch -> th ) $.
    $( Conjoin antecedents and consequents of two premises.  (Contributed by
       NM, 3-Jan-1993.)  (Proof shortened by Wolf Lammen, 14-Dec-2013.) $)
    anim12i $p |- ( ( ph /\ ch ) -> ( ps /\ th ) ) $=
      ( wa id syl2an ) ABDBDGZCEFJHI $.

    $( Variant of ~ anim12i with commutation.  (Contributed by Jonathan
       Ben-Naim, 3-Jun-2011.) $)
    anim12ci $p |- ( ( ph /\ ch ) -> ( th /\ ps ) ) $=
      ( wa anim12i ancoms ) CADBGCDABFEHI $.
  $}

  ${
    anim1i.1 $e |- ( ph -> ps ) $.
    $( Introduce conjunct to both sides of an implication.  (Contributed by NM,
       5-Aug-1993.) $)
    anim1i $p |- ( ( ph /\ ch ) -> ( ps /\ ch ) ) $=
      ( id anim12i ) ABCCDCEF $.

    $( Introduce conjunct to both sides of an implication.  (Contributed by NM,
       3-Jan-1993.) $)
    anim2i $p |- ( ( ch /\ ph ) -> ( ch /\ ps ) ) $=
      ( id anim12i ) CCABCEDF $.
  $}

  ${
    anim12ii.1 $e |- ( ph -> ( ps -> ch ) ) $.
    anim12ii.2 $e |- ( th -> ( ps -> ta ) ) $.
    $( Conjoin antecedents and consequents in a deduction.  (Contributed by NM,
       11-Nov-2007.)  (Proof shortened by Wolf Lammen, 19-Jul-2013.) $)
    anim12ii $p |- ( ( ph /\ th ) -> ( ps -> ( ch /\ ta ) ) ) $=
      ( wa wi adantr adantl jcad ) ADHBCEABCIDFJDBEIAGKL $.
  $}

  $( Conjoin antecedents and consequents of two premises.  This is the closed
     theorem form of ~ anim12d .  Theorem *3.47 of [WhiteheadRussell] p. 113.
     It was proved by Leibniz, and it evidently pleased him enough to call it
     _praeclarum theorema_ (splendid theorem).  (Contributed by NM,
     12-Aug-1993.)  (Proof shortened by Wolf Lammen, 7-Apr-2013.) $)
  prth $p |- ( ( ( ph -> ps ) /\ ( ch -> th ) )
       -> ( ( ph /\ ch ) -> ( ps /\ th ) ) ) $=
    ( wi wa simpl simpr anim12d ) ABEZCDEZFABCDJKGJKHI $.

  $( Theorem *2.3 of [WhiteheadRussell] p. 104.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.3 $p |- ( ( ph \/ ( ps \/ ch ) ) -> ( ph \/ ( ch \/ ps ) ) ) $=
    ( wo pm1.4 orim2i ) BCDCBDABCEF $.

  $( Theorem *2.41 of [WhiteheadRussell] p. 106.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.41 $p |- ( ( ps \/ ( ph \/ ps ) ) -> ( ph \/ ps ) ) $=
    ( wo olc id jaoi ) BABCZGBADGEF $.

  $( Theorem *2.42 of [WhiteheadRussell] p. 106.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.42 $p |- ( ( -. ph \/ ( ph -> ps ) ) -> ( ph -> ps ) ) $=
    ( wn wi pm2.21 id jaoi ) ACABDZHABEHFG $.

  $( Theorem *2.4 of [WhiteheadRussell] p. 106.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.4 $p |- ( ( ph \/ ( ph \/ ps ) ) -> ( ph \/ ps ) ) $=
    ( wo orc id jaoi ) AABCZGABDGEF $.

  ${
    pm2.65da.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    pm2.65da.2 $e |- ( ( ph /\ ps ) -> -. ch ) $.
    $( Deduction rule for proof by contradiction.  (Contributed by NM,
       12-Jun-2014.) $)
    pm2.65da $p |- ( ph -> -. ps ) $=
      ( ex wn pm2.65d ) ABCABCDFABCGEFH $.
  $}

  $( Theorem *4.44 of [WhiteheadRussell] p. 119.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.44 $p |- ( ph <-> ( ph \/ ( ph /\ ps ) ) ) $=
    ( wa wo orc id simpl jaoi impbii ) AAABCZDAJEAAJAFABGHI $.

  $( Theorem *4.14 of [WhiteheadRussell] p. 117.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 23-Oct-2012.) $)
  pm4.14 $p |- ( ( ( ph /\ ps ) -> ch ) <-> ( ( ph /\ -. ch ) -> -. ps ) ) $=
    ( wi wn wa con34b imbi2i impexp 3bitr4i ) ABCDZDACEZBEZDZDABFCDALFMDKNABCGH
    ABCIALMIJ $.

  $( Theorem *3.37 (Transp) of [WhiteheadRussell] p. 112.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 23-Oct-2012.) $)
  pm3.37 $p |- ( ( ( ph /\ ps ) -> ch ) -> ( ( ph /\ -. ch ) -> -. ps ) ) $=
    ( wa wi wn pm4.14 biimpi ) ABDCEACFDBFEABCGH $.

  $( Theorem to move a conjunct in and out of a negation.  (Contributed by NM,
     9-Nov-2003.) $)
  nan $p |- ( ( ph -> -. ( ps /\ ch ) ) <-> ( ( ph /\ ps ) -> -. ch ) ) $=
    ( wa wn wi impexp imnan imbi2i bitr2i ) ABDCEZFABKFZFABCDEZFABKGLMABCHIJ $.

  $( Theorem *4.15 of [WhiteheadRussell] p. 117.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 18-Nov-2012.) $)
  pm4.15 $p |- ( ( ( ph /\ ps ) -> -. ch ) <-> ( ( ps /\ ch ) -> -. ph ) ) $=
    ( wa wn wi con2b nan bitr2i ) BCDZAEFAJEFABDCEFJAGABCHI $.

  $( Implication distributes over disjunction.  Theorem *4.78 of
     [WhiteheadRussell] p. 121.  (Contributed by NM, 3-Jan-2005.)  (Proof
     shortened by Wolf Lammen, 19-Nov-2012.) $)
  pm4.78 $p |- ( ( ( ph -> ps ) \/ ( ph -> ch ) ) <->
                ( ph -> ( ps \/ ch ) ) ) $=
    ( wn wo wi orordi imor orbi12i 3bitr4ri ) ADZBCEZEKBEZKCEZEALFABFZACFZEKBCG
    ALHOMPNABHACHIJ $.

  $( Theorem *4.79 of [WhiteheadRussell] p. 121.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 27-Jun-2013.) $)
  pm4.79 $p |- ( ( ( ps -> ph ) \/ ( ch -> ph ) ) <->
                ( ( ps /\ ch ) -> ph ) ) $=
    ( wi wo wa id jaoa wn simplim pm3.3 syl5 orrd impbii ) BADZCADZEBCFADZOBAPC
    OGPGHQOPOIBQPBAJBCAKLMN $.

  $( Theorem *4.87 of [WhiteheadRussell] p. 122.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Eric Schmidt, 26-Oct-2006.) $)
  pm4.87 $p |- ( ( ( ( ( ph /\ ps ) -> ch ) <-> ( ph -> ( ps -> ch ) ) ) /\
                ( ( ph -> ( ps -> ch ) ) <-> ( ps -> ( ph -> ch ) ) ) ) /\
                ( ( ps -> ( ph -> ch ) ) <-> ( ( ps /\ ph ) -> ch ) ) ) $=
    ( wa wi wb impexp bi2.04 pm3.2i bicomi ) ABDCEABCEEZFZKBACEEZFZDMBADCEZFLNA
    BCGABCHIOMBACGJI $.

  $( Theorem *3.33 (Syll) of [WhiteheadRussell] p. 112.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.33 $p |- ( ( ( ph -> ps ) /\ ( ps -> ch ) ) -> ( ph -> ch ) ) $=
    ( wi imim1 imp ) ABDBCDACDABCEF $.

  $( Theorem *3.34 (Syll) of [WhiteheadRussell] p. 112.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.34 $p |- ( ( ( ps -> ch ) /\ ( ph -> ps ) ) -> ( ph -> ch ) ) $=
    ( wi imim2 imp ) BCDABDACDBCAEF $.

  $( Conjunctive detachment.  Theorem *3.35 of [WhiteheadRussell] p. 112.
     (Contributed by NM, 14-Dec-2002.) $)
  pm3.35 $p |- ( ( ph /\ ( ph -> ps ) ) -> ps ) $=
    ( wi pm2.27 imp ) AABCBABDE $.

  $( Theorem *5.31 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.31 $p |- ( ( ch /\ ( ph -> ps ) ) -> ( ph -> ( ps /\ ch ) ) ) $=
    ( wi wa pm3.21 imim2d imp ) CABDABCEZDCBIACBFGH $.

  ${
    imp4.1 $e |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $.
    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp4a $p |- ( ph -> ( ps -> ( ( ch /\ th ) -> ta ) ) ) $=
      ( wi wa impexp syl6ibr ) ABCDEGGCDHEGFCDEIJ $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp4b $p |- ( ( ph /\ ps ) -> ( ( ch /\ th ) -> ta ) ) $=
      ( wa wi imp4a imp ) ABCDGEHABCDEFIJ $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp4c $p |- ( ph -> ( ( ( ps /\ ch ) /\ th ) -> ta ) ) $=
      ( wa wi impd ) ABCGDEABCDEHFII $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp4d $p |- ( ph -> ( ( ps /\ ( ch /\ th ) ) -> ta ) ) $=
      ( wa imp4a impd ) ABCDGEABCDEFHI $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp41 $p |- ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) -> ta ) $=
      ( wa wi imp imp31 ) ABGCDEABCDEHHFIJ $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp42 $p |- ( ( ( ph /\ ( ps /\ ch ) ) /\ th ) -> ta ) $=
      ( wa wi imp32 imp ) ABCGGDEABCDEHFIJ $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp43 $p |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) -> ta ) $=
      ( wa imp4b imp ) ABGCDGEABCDEFHI $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp44 $p |- ( ( ph /\ ( ( ps /\ ch ) /\ th ) ) -> ta ) $=
      ( wa imp4c imp ) ABCGDGEABCDEFHI $.

    $( An importation inference.  (Contributed by NM, 26-Apr-1994.) $)
    imp45 $p |- ( ( ph /\ ( ps /\ ( ch /\ th ) ) ) -> ta ) $=
      ( wa imp4d imp ) ABCDGGEABCDEFHI $.
  $}

  ${
    imp5.1 $e |- ( ph -> ( ps -> ( ch -> ( th -> ( ta -> et ) ) ) ) ) $.
    $( An importation inference.  (Contributed by Jeff Hankins, 7-Jul-2009.) $)
    imp5a $p |- ( ph -> ( ps -> ( ch -> ( ( th /\ ta ) -> et ) ) ) ) $=
      ( wi wa pm3.31 syl8 ) ABCDEFHHDEIFHGDEFJK $.

    $( An importation inference.  (Contributed by Jeff Hankins, 7-Jul-2009.) $)
    imp5d $p |- ( ( ( ph /\ ps ) /\ ch ) -> ( ( th /\ ta ) -> et ) ) $=
      ( wa wi imp31 impd ) ABHCHDEFABCDEFIIGJK $.

    $( An importation inference.  (Contributed by Jeff Hankins, 7-Jul-2009.) $)
    imp5g $p |- ( ( ph /\ ps ) -> ( ( ( ch /\ th ) /\ ta ) -> et ) ) $=
      ( wa wi imp imp4c ) ABHCDEFABCDEFIIIGJK $.

    $( An importation inference.  (Contributed by Jeff Hankins, 7-Jul-2009.) $)
    imp55 $p |- ( ( ( ph /\ ( ps /\ ( ch /\ th ) ) ) /\ ta ) -> et ) $=
      ( wa wi imp4a imp42 ) ABCDHEFABCDEFIGJK $.

    $( An importation inference.  (Contributed by Jeff Hankins, 7-Jul-2009.) $)
    imp511 $p |- ( ( ph /\ ( ( ps /\ ( ch /\ th ) ) /\ ta ) ) -> et ) $=
      ( wa wi imp4a imp44 ) ABCDHEFABCDEFIGJK $.
  $}

  ${
    expimpd.1 $e |- ( ( ph /\ ps ) -> ( ch -> th ) ) $.
    $( Exportation followed by a deduction version of importation.
       (Contributed by NM, 6-Sep-2008.) $)
    expimpd $p |- ( ph -> ( ( ps /\ ch ) -> th ) ) $=
      ( wi ex impd ) ABCDABCDFEGH $.
  $}

  ${
    exp31.1 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp31 $p |- ( ph -> ( ps -> ( ch -> th ) ) ) $=
      ( wi wa ex ) ABCDFABGCDEHH $.
  $}

  ${
    exp32.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp32 $p |- ( ph -> ( ps -> ( ch -> th ) ) ) $=
      ( wa ex expd ) ABCDABCFDEGH $.
  $}

  ${
    exp4a.1 $e |- ( ph -> ( ps -> ( ( ch /\ th ) -> ta ) ) ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp4a $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wa wi impexp syl6ib ) ABCDGEHCDEHHFCDEIJ $.
  $}

  ${
    exp4b.1 $e |- ( ( ph /\ ps ) -> ( ( ch /\ th ) -> ta ) ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.)  (Proof
       shortened by Wolf Lammen, 23-Nov-2012.) $)
    exp4b $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wa wi ex exp4a ) ABCDEABCDGEHFIJ $.
  $}

  ${
    exp4c.1 $e |- ( ph -> ( ( ( ps /\ ch ) /\ th ) -> ta ) ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp4c $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wi wa expd ) ABCDEGABCHDEFII $.
  $}

  ${
    exp4d.1 $e |- ( ph -> ( ( ps /\ ( ch /\ th ) ) -> ta ) ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp4d $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wa expd exp4a ) ABCDEABCDGEFHI $.
  $}

  ${
    exp41.1 $e |- ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) -> ta ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp41 $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wi wa ex exp31 ) ABCDEGABHCHDEFIJ $.
  $}

  ${
    exp42.1 $e |- ( ( ( ph /\ ( ps /\ ch ) ) /\ th ) -> ta ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp42 $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wi wa exp31 expd ) ABCDEGABCHDEFIJ $.
  $}

  ${
    exp43.1 $e |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) -> ta ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp43 $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wa ex exp4b ) ABCDEABGCDGEFHI $.
  $}

  ${
    exp44.1 $e |- ( ( ph /\ ( ( ps /\ ch ) /\ th ) ) -> ta ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp44 $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wi wa exp32 expd ) ABCDEGABCHDEFIJ $.
  $}

  ${
    exp45.1 $e |- ( ( ph /\ ( ps /\ ( ch /\ th ) ) ) -> ta ) $.
    $( An exportation inference.  (Contributed by NM, 26-Apr-1994.) $)
    exp45 $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wa exp32 exp4a ) ABCDEABCDGEFHI $.
  $}

  ${
    expr.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( Export a wff from a right conjunct.  (Contributed by Jeff Hankins,
       30-Aug-2009.) $)
    expr $p |- ( ( ph /\ ps ) -> ( ch -> th ) ) $=
      ( wi exp32 imp ) ABCDFABCDEGH $.
  $}

  ${
    exp5c.1 $e |- ( ph -> ( ( ps /\ ch ) -> ( ( th /\ ta ) -> et ) ) ) $.
    $( An exportation inference.  (Contributed by Jeff Hankins, 7-Jul-2009.) $)
    exp5c $p |- ( ph -> ( ps -> ( ch -> ( th -> ( ta -> et ) ) ) ) ) $=
      ( wi wa exp4a expd ) ABCDEFHHABCIDEFGJK $.
  $}

  ${
    exp53.1 $e |- ( ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) /\ ta ) -> et ) $.
    $( An exportation inference.  (Contributed by Jeff Hankins,
       30-Aug-2009.) $)
    exp53 $p |- ( ph -> ( ps -> ( ch -> ( th -> ( ta -> et ) ) ) ) ) $=
      ( wi wa ex exp43 ) ABCDEFHABICDIIEFGJK $.
  $}

  ${
    expl.1 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( Export a wff from a left conjunct.  (Contributed by Jeff Hankins,
       28-Aug-2009.) $)
    expl $p |- ( ph -> ( ( ps /\ ch ) -> th ) ) $=
      ( exp31 impd ) ABCDABCDEFG $.
  $}

  ${
    impr.1 $e |- ( ( ph /\ ps ) -> ( ch -> th ) ) $.
    $( Import a wff into a right conjunct.  (Contributed by Jeff Hankins,
       30-Aug-2009.) $)
    impr $p |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $=
      ( wi ex imp32 ) ABCDABCDFEGH $.
  $}

  ${
    impl.1 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( Export a wff from a left conjunct.  (Contributed by Mario Carneiro,
       9-Jul-2014.) $)
    impl $p |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $=
      ( expd imp31 ) ABCDABCDEFG $.
  $}

  ${
    impac.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Importation with conjunction in consequent.  (Contributed by NM,
       9-Aug-1994.) $)
    impac $p |- ( ( ph /\ ps ) -> ( ch /\ ps ) ) $=
      ( wa ancrd imp ) ABCBEABCDFG $.
  $}

  ${
    exbiri.1 $e |- ( ( ph /\ ps ) -> ( ch <-> th ) ) $.
    $( Inference form of ~ exbir .  This proof is ~ exbiriVD automatically
       translated and minimized.  (Contributed by Alan Sare, 31-Dec-2011.)
       (Proof shortened by Wolf Lammen, 27-Jan-2013.) $)
    exbiri $p |- ( ph -> ( ps -> ( th -> ch ) ) ) $=
      ( wa biimpar exp31 ) ABDCABFCDEGH $.
  $}

  ${
    pm3.26bda.1 $e |- ( ph -> ( ps <-> ( ch /\ th ) ) ) $.
    $( Deduction eliminating a conjunct.  (Contributed by NM, 22-Oct-2007.) $)
    simprbda $p |- ( ( ph /\ ps ) -> ch ) $=
      ( wa biimpa simpld ) ABFCDABCDFEGH $.

    $( Deduction eliminating a conjunct.  (Contributed by NM, 22-Oct-2007.) $)
    simplbda $p |- ( ( ph /\ ps ) -> th ) $=
      ( wa biimpa simprd ) ABFCDABCDFEGH $.
  $}

  ${
    simplbi2.1 $e |- ( ph <-> ( ps /\ ch ) ) $.
    $( Deduction eliminating a conjunct.  (Contributed by Alan Sare,
       31-Dec-2011.) $)
    simplbi2 $p |- ( ps -> ( ch -> ph ) ) $=
      ( wa biimpri ex ) BCAABCEDFG $.
  $}

  $( Closed form of ~ simplbi2com .  (Contributed by Alan Sare,
     22-Jul-2012.) $)
  simplbi2comt $p |- ( ( ph <-> ( ps /\ ch ) ) -> ( ch -> ( ps -> ph ) ) ) $=
    ( wa wb bi2 expcomd ) ABCDZEBCAAHFG $.

  ${
    simplbi2com.1 $e |- ( ph <-> ( ps /\ ch ) ) $.
    $( A deduction eliminating a conjunct, similar to ~ simplbi2 .
       (Contributed by Alan Sare, 22-Jul-2012.)  (Proof shortened by Wolf
       Lammen, 10-Nov-2012.) $)
    simplbi2com $p |- ( ch -> ( ps -> ph ) ) $=
      ( simplbi2 com12 ) BCAABCDEF $.
  $}

  $( A theorem similar to the standard definition of the biconditional.
     Definition of [Margaris] p. 49.  (Contributed by NM, 24-Jan-1993.) $)
  dfbi2 $p |- ( ( ph <-> ps ) <-> ( ( ph -> ps ) /\ ( ps -> ph ) ) ) $=
    ( wb wi wn wa dfbi1 df-an bitr4i ) ABCABDZBADZEDEJKFABGJKHI $.

  $( Definition ~ df-bi rewritten in an abbreviated form to help intuitive
     understanding of that definition.  Note that it is a conjunction of two
     implications; one which asserts properties that follow from the
     biconditional and one which asserts properties that imply the
     biconditional.  (Contributed by NM, 15-Aug-2008.) $)
  dfbi $p |- ( ( ( ph <-> ps ) -> ( ( ph -> ps ) /\ ( ps -> ph ) ) )
        /\ ( ( ( ph -> ps ) /\ ( ps -> ph ) ) -> ( ph <-> ps ) ) ) $=
    ( wb wi wa dfbi2 biimpi biimpri pm3.2i ) ABCZABDBADEZDKJDJKABFZGJKLHI $.

  $( Implication in terms of biconditional and conjunction.  Theorem *4.71 of
     [WhiteheadRussell] p. 120.  (Contributed by NM, 21-Jun-1993.)  (Proof
     shortened by Wolf Lammen, 2-Dec-2012.) $)
  pm4.71 $p |- ( ( ph -> ps ) <-> ( ph <-> ( ph /\ ps ) ) ) $=
    ( wa wi wb simpl biantru anclb dfbi2 3bitr4i ) AABCZDZLKADZCABDAKEMLABFGABH
    AKIJ $.

  $( Implication in terms of biconditional and conjunction.  Theorem *4.71 of
     [WhiteheadRussell] p. 120 (with conjunct reversed).  (Contributed by NM,
     25-Jul-1999.) $)
  pm4.71r $p |- ( ( ph -> ps ) <-> ( ph <-> ( ps /\ ph ) ) ) $=
    ( wi wa wb pm4.71 ancom bibi2i bitri ) ABCAABDZEABADZEABFJKAABGHI $.

  ${
    pm4.71i.1 $e |- ( ph -> ps ) $.
    $( Inference converting an implication to a biconditional with
       conjunction.  Inference from Theorem *4.71 of [WhiteheadRussell]
       p. 120.  (Contributed by NM, 4-Jan-2004.) $)
    pm4.71i $p |- ( ph <-> ( ph /\ ps ) ) $=
      ( wi wa wb pm4.71 mpbi ) ABDAABEFCABGH $.
  $}

  ${
    pm4.71ri.1 $e |- ( ph -> ps ) $.
    $( Inference converting an implication to a biconditional with
       conjunction.  Inference from Theorem *4.71 of [WhiteheadRussell] p. 120
       (with conjunct reversed).  (Contributed by NM, 1-Dec-2003.) $)
    pm4.71ri $p |- ( ph <-> ( ps /\ ph ) ) $=
      ( wi wa wb pm4.71r mpbi ) ABDABAEFCABGH $.
  $}

  ${
    pm4.71rd.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Deduction converting an implication to a biconditional with
       conjunction.  Deduction from Theorem *4.71 of [WhiteheadRussell]
       p. 120.  (Contributed by Mario Carneiro, 25-Dec-2016.) $)
    pm4.71d $p |- ( ph -> ( ps <-> ( ps /\ ch ) ) ) $=
      ( wi wa wb pm4.71 sylib ) ABCEBBCFGDBCHI $.

    $( Deduction converting an implication to a biconditional with
       conjunction.  Deduction from Theorem *4.71 of [WhiteheadRussell]
       p. 120.  (Contributed by NM, 10-Feb-2005.) $)
    pm4.71rd $p |- ( ph -> ( ps <-> ( ch /\ ps ) ) ) $=
      ( wi wa wb pm4.71r sylib ) ABCEBCBFGDBCHI $.
  $}

  $( Distribution of implication over biconditional.  Theorem *5.32 of
     [WhiteheadRussell] p. 125.  (Contributed by NM, 1-Aug-1994.) $)
  pm5.32 $p |- ( ( ph -> ( ps <-> ch ) ) <->
               ( ( ph /\ ps ) <-> ( ph /\ ch ) ) ) $=
    ( wb wi wn wa notbi imbi2i pm5.74 3bitri df-an bibi12i bitr4i ) ABCDZEZABFZ
    EZFZACFZEZFZDZABGZACGZDPAQTDZERUADUCOUFABCHIAQTJRUAHKUDSUEUBABLACLMN $.

  ${
    pm5.32i.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Distribution of implication over biconditional (inference rule).
       (Contributed by NM, 1-Aug-1994.) $)
    pm5.32i $p |- ( ( ph /\ ps ) <-> ( ph /\ ch ) ) $=
      ( wb wi wa pm5.32 mpbi ) ABCEFABGACGEDABCHI $.

    $( Distribution of implication over biconditional (inference rule).
       (Contributed by NM, 12-Mar-1995.) $)
    pm5.32ri $p |- ( ( ps /\ ph ) <-> ( ch /\ ph ) ) $=
      ( wa pm5.32i ancom 3bitr4i ) ABEACEBAECAEABCDFBAGCAGH $.
  $}

  ${
    pm5.32d.1 $e |- ( ph -> ( ps -> ( ch <-> th ) ) ) $.
    $( Distribution of implication over biconditional (deduction rule).
       (Contributed by NM, 29-Oct-1996.) $)
    pm5.32d $p |- ( ph -> ( ( ps /\ ch ) <-> ( ps /\ th ) ) ) $=
      ( wb wi wa pm5.32 sylib ) ABCDFGBCHBDHFEBCDIJ $.

    $( Distribution of implication over biconditional (deduction rule).
       (Contributed by NM, 25-Dec-2004.) $)
    pm5.32rd $p |- ( ph -> ( ( ch /\ ps ) <-> ( th /\ ps ) ) ) $=
      ( wa pm5.32d ancom 3bitr4g ) ABCFBDFCBFDBFABCDEGCBHDBHI $.
  $}

  ${
    pm5.32da.1 $e |- ( ( ph /\ ps ) -> ( ch <-> th ) ) $.
    $( Distribution of implication over biconditional (deduction rule).
       (Contributed by NM, 9-Dec-2006.) $)
    pm5.32da $p |- ( ph -> ( ( ps /\ ch ) <-> ( ps /\ th ) ) ) $=
      ( wb ex pm5.32d ) ABCDABCDFEGH $.
  $}

  ${
    biadan2.1 $e |- ( ph -> ps ) $.
    biadan2.2 $e |- ( ps -> ( ph <-> ch ) ) $.
    $( Add a conjunction to an equivalence.  (Contributed by Jeff Madsen,
       20-Jun-2011.) $)
    biadan2 $p |- ( ph <-> ( ps /\ ch ) ) $=
      ( wa pm4.71ri pm5.32i bitri ) ABAFBCFABDGBACEHI $.
  $}

  $( Theorem *4.24 of [WhiteheadRussell] p. 117.  (Contributed by NM,
     11-May-1993.) $)
  pm4.24 $p |- ( ph <-> ( ph /\ ph ) ) $=
    ( id pm4.71i ) AAABC $.

  $( Idempotent law for conjunction.  (Contributed by NM, 8-Jan-2004.)  (Proof
     shortened by Wolf Lammen, 14-Mar-2014.) $)
  anidm $p |- ( ( ph /\ ph ) <-> ph ) $=
    ( wa pm4.24 bicomi ) AAABACD $.

  ${
    anidms.1 $e |- ( ( ph /\ ph ) -> ps ) $.
    $( Inference from idempotent law for conjunction.  (Contributed by NM,
       15-Jun-1994.) $)
    anidms $p |- ( ph -> ps ) $=
      ( ex pm2.43i ) ABAABCDE $.
  $}

  $( Conjunction idempotence with antecedent.  (Contributed by Roy F. Longton,
     8-Aug-2005.) $)
  anidmdbi $p |- ( ( ph -> ( ps /\ ps ) ) <-> ( ph -> ps ) ) $=
    ( wa anidm imbi2i ) BBCBABDE $.

  ${
    anasss.1 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( Associative law for conjunction applied to antecedent (eliminates
       syllogism).  (Contributed by NM, 15-Nov-2002.) $)
    anasss $p |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $=
      ( exp31 imp32 ) ABCDABCDEFG $.
  $}

  ${
    anassrs.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( Associative law for conjunction applied to antecedent (eliminates
       syllogism).  (Contributed by NM, 15-Nov-2002.) $)
    anassrs $p |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $=
      ( exp32 imp31 ) ABCDABCDEFG $.
  $}

  $( Associative law for conjunction.  Theorem *4.32 of [WhiteheadRussell]
     p. 118.  (Contributed by NM, 21-Jun-1993.)  (Proof shortened by Wolf
     Lammen, 24-Nov-2012.) $)
  anass $p |- ( ( ( ph /\ ps ) /\ ch ) <-> ( ph /\ ( ps /\ ch ) ) ) $=
    ( wa id anassrs anasss impbii ) ABDCDZABCDDZABCJJEFABCIIEGH $.

  ${
    sylanl1.1 $e |- ( ph -> ps ) $.
    sylanl1.2 $e |- ( ( ( ps /\ ch ) /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 10-Mar-2005.) $)
    sylanl1 $p |- ( ( ( ph /\ ch ) /\ th ) -> ta ) $=
      ( wa anim1i sylan ) ACHBCHDEABCFIGJ $.
  $}

  ${
    sylanl2.1 $e |- ( ph -> ch ) $.
    sylanl2.2 $e |- ( ( ( ps /\ ch ) /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 1-Jan-2005.) $)
    sylanl2 $p |- ( ( ( ps /\ ph ) /\ th ) -> ta ) $=
      ( wa anim2i sylan ) BAHBCHDEACBFIGJ $.
  $}

  ${
    sylanr1.1 $e |- ( ph -> ch ) $.
    sylanr1.2 $e |- ( ( ps /\ ( ch /\ th ) ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 9-Apr-2005.) $)
    sylanr1 $p |- ( ( ps /\ ( ph /\ th ) ) -> ta ) $=
      ( wa anim1i sylan2 ) ADHBCDHEACDFIGJ $.
  $}

  ${
    sylanr2.1 $e |- ( ph -> th ) $.
    sylanr2.2 $e |- ( ( ps /\ ( ch /\ th ) ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 9-Apr-2005.) $)
    sylanr2 $p |- ( ( ps /\ ( ch /\ ph ) ) -> ta ) $=
      ( wa anim2i sylan2 ) CAHBCDHEADCFIGJ $.
  $}

  ${
    sylani.1 $e |- ( ph -> ch ) $.
    sylani.2 $e |- ( ps -> ( ( ch /\ th ) -> ta ) ) $.
    $( A syllogism inference.  (Contributed by NM, 2-May-1996.) $)
    sylani $p |- ( ps -> ( ( ph /\ th ) -> ta ) ) $=
      ( wi a1i syland ) BACDEACHBFIGJ $.
  $}

  ${
    sylan2i.1 $e |- ( ph -> th ) $.
    sylan2i.2 $e |- ( ps -> ( ( ch /\ th ) -> ta ) ) $.
    $( A syllogism inference.  (Contributed by NM, 1-Aug-1994.) $)
    sylan2i $p |- ( ps -> ( ( ch /\ ph ) -> ta ) ) $=
      ( wi a1i sylan2d ) BADCEADHBFIGJ $.
  $}

  ${
    syl2ani.1 $e |- ( ph -> ch ) $.
    syl2ani.2 $e |- ( et -> th ) $.
    syl2ani.3 $e |- ( ps -> ( ( ch /\ th ) -> ta ) ) $.
    $( A syllogism inference.  (Contributed by NM, 3-Aug-1999.) $)
    syl2ani $p |- ( ps -> ( ( ph /\ et ) -> ta ) ) $=
      ( sylan2i sylani ) ABCFEGFBCDEHIJK $.
  $}

  ${
    sylan9.1 $e |- ( ph -> ( ps -> ch ) ) $.
    sylan9.2 $e |- ( th -> ( ch -> ta ) ) $.
    $( Nested syllogism inference conjoining dissimilar antecedents.
       (Contributed by NM, 14-May-1993.)  (Proof shortened by Andrew Salmon,
       7-May-2011.) $)
    sylan9 $p |- ( ( ph /\ th ) -> ( ps -> ta ) ) $=
      ( wi syl9 imp ) ADBEHABCDEFGIJ $.
  $}

  ${
    sylan9r.1 $e |- ( ph -> ( ps -> ch ) ) $.
    sylan9r.2 $e |- ( th -> ( ch -> ta ) ) $.
    $( Nested syllogism inference conjoining dissimilar antecedents.
       (Contributed by NM, 14-May-1993.) $)
    sylan9r $p |- ( ( th /\ ph ) -> ( ps -> ta ) ) $=
      ( wi syl9r imp ) DABEHABCDEFGIJ $.
  $}

  ${
    mtand.1 $e |- ( ph -> -. ch ) $.
    mtand.2 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( A modus tollens deduction.  (Contributed by Jeff Hankins,
       19-Aug-2009.) $)
    mtand $p |- ( ph -> -. ps ) $=
      ( ex mtod ) ABCDABCEFG $.
  $}

  ${
    mtord.1 $e |- ( ph -> -. ch ) $.
    mtord.2 $e |- ( ph -> -. th ) $.
    mtord.3 $e |- ( ph -> ( ps -> ( ch \/ th ) ) ) $.
    $( A modus tollens deduction involving disjunction.  (Contributed by Jeff
       Hankins, 15-Jul-2009.) $)
    mtord $p |- ( ph -> -. ps ) $=
      ( wn wo wi df-or syl6ib mpid mtod ) ABDFABCHZDEABCDIODJGCDKLMN $.
  $}

  ${
    syl2anc.1 $e |- ( ph -> ps ) $.
    syl2anc.2 $e |- ( ph -> ch ) $.
    syl2anc.3 $e |- ( ( ps /\ ch ) -> th ) $.
    $( Syllogism inference combined with contraction.  (Contributed by NM,
       16-Mar-2012.) $)
    syl2anc $p |- ( ph -> th ) $=
      ( ex sylc ) ABCDEFBCDGHI $.
  $}

  ${
    sylancl.1 $e |- ( ph -> ps ) $.
    sylancl.2 $e |- ch $.
    sylancl.3 $e |- ( ( ps /\ ch ) -> th ) $.
    $( Syllogism inference combined with modus ponens.  (Contributed by Jeff
       Madsen, 2-Sep-2009.) $)
    sylancl $p |- ( ph -> th ) $=
      ( a1i syl2anc ) ABCDECAFHGI $.
  $}

  ${
    sylancr.1 $e |- ps $.
    sylancr.2 $e |- ( ph -> ch ) $.
    sylancr.3 $e |- ( ( ps /\ ch ) -> th ) $.
    $( Syllogism inference combined with modus ponens.  (Contributed by Jeff
       Madsen, 2-Sep-2009.) $)
    sylancr $p |- ( ph -> th ) $=
      ( a1i syl2anc ) ABCDBAEHFGI $.
  $}

  ${
    sylanbrc.1 $e |- ( ph -> ps ) $.
    sylanbrc.2 $e |- ( ph -> ch ) $.
    sylanbrc.3 $e |- ( th <-> ( ps /\ ch ) ) $.
    $( Syllogism inference.  (Contributed by Jeff Madsen, 2-Sep-2009.) $)
    sylanbrc $p |- ( ph -> th ) $=
      ( wa jca sylibr ) ABCHDABCEFIGJ $.
  $}

  ${
    sylancb.1 $e |- ( ph <-> ps ) $.
    sylancb.2 $e |- ( ph <-> ch ) $.
    sylancb.3 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A syllogism inference combined with contraction.  (Contributed by NM,
       3-Sep-2004.) $)
    sylancb $p |- ( ph -> th ) $=
      ( syl2anb anidms ) ADABCDAEFGHI $.
  $}

  ${
    sylancbr.1 $e |- ( ps <-> ph ) $.
    sylancbr.2 $e |- ( ch <-> ph ) $.
    sylancbr.3 $e |- ( ( ps /\ ch ) -> th ) $.
    $( A syllogism inference combined with contraction.  (Contributed by NM,
       3-Sep-2004.) $)
    sylancbr $p |- ( ph -> th ) $=
      ( syl2anbr anidms ) ADABCDAEFGHI $.
  $}

  ${
    sylancom.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    sylancom.2 $e |- ( ( ch /\ ps ) -> th ) $.
    $( Syllogism inference with commutation of antecedents.  (Contributed by
       NM, 2-Jul-2008.) $)
    sylancom $p |- ( ( ph /\ ps ) -> th ) $=
      ( wa simpr syl2anc ) ABGCBDEABHFI $.
  $}

  ${
    mpdan.1 $e |- ( ph -> ps ) $.
    mpdan.2 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 23-May-1999.)
       (Proof shortened by Wolf Lammen, 22-Nov-2012.) $)
    mpdan $p |- ( ph -> ch ) $=
      ( id syl2anc ) AABCAFDEG $.
  $}

  ${
    mpancom.1 $e |- ( ps -> ph ) $.
    mpancom.2 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( An inference based on modus ponens with commutation of antecedents.
       (Contributed by NM, 28-Oct-2003.)  (Proof shortened by Wolf Lammen,
       7-Apr-2013.) $)
    mpancom $p |- ( ps -> ch ) $=
      ( id syl2anc ) BABCDBFEG $.
  $}

  ${
    mpan.1 $e |- ph $.
    mpan.2 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 30-Aug-1993.)
       (Proof shortened by Wolf Lammen, 7-Apr-2013.) $)
    mpan $p |- ( ps -> ch ) $=
      ( a1i mpancom ) ABCABDFEG $.
  $}

  ${
    mpan2.1 $e |- ps $.
    mpan2.2 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 16-Sep-1993.)
       (Proof shortened by Wolf Lammen, 19-Nov-2012.) $)
    mpan2 $p |- ( ph -> ch ) $=
      ( a1i mpdan ) ABCBADFEG $.
  $}

  ${
    mp2an.1 $e |- ph $.
    mp2an.2 $e |- ps $.
    mp2an.3 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       13-Apr-1995.) $)
    mp2an $p |- ch $=
      ( mpan ax-mp ) BCEABCDFGH $.
  $}

  ${
    mp4an.1 $e |- ph $.
    mp4an.2 $e |- ps $.
    mp4an.3 $e |- ch $.
    mp4an.4 $e |- th $.
    mp4an.5 $e |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) -> ta ) $.
    $( An inference based on modus ponens.  (Contributed by Jeff Madsen,
       15-Jun-2010.) $)
    mp4an $p |- ta $=
      ( wa pm3.2i mp2an ) ABKCDKEABFGLCDHILJM $.
  $}

  ${
    mpan2d.1 $e |- ( ph -> ch ) $.
    mpan2d.2 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( A deduction based on modus ponens.  (Contributed by NM, 12-Dec-2004.) $)
    mpan2d $p |- ( ph -> ( ps -> th ) ) $=
      ( expd mpid ) ABCDEABCDFGH $.
  $}

  ${
    mpand.1 $e |- ( ph -> ps ) $.
    mpand.2 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( A deduction based on modus ponens.  (Contributed by NM, 12-Dec-2004.)
       (Proof shortened by Wolf Lammen, 7-Apr-2013.) $)
    mpand $p |- ( ph -> ( ch -> th ) ) $=
      ( ancomsd mpan2d ) ACBDEABCDFGH $.
  $}

  ${
    mpani.1 $e |- ps $.
    mpani.2 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 10-Apr-1994.)
       (Proof shortened by Wolf Lammen, 19-Nov-2012.) $)
    mpani $p |- ( ph -> ( ch -> th ) ) $=
      ( a1i mpand ) ABCDBAEGFH $.
  $}

  ${
    mpan2i.1 $e |- ch $.
    mpan2i.2 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 10-Apr-1994.)
       (Proof shortened by Wolf Lammen, 19-Nov-2012.) $)
    mpan2i $p |- ( ph -> ( ps -> th ) ) $=
      ( a1i mpan2d ) ABCDCAEGFH $.
  $}

  ${
    mp2ani.1 $e |- ps $.
    mp2ani.2 $e |- ch $.
    mp2ani.3 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       12-Dec-2004.) $)
    mp2ani $p |- ( ph -> th ) $=
      ( mpani mpi ) ACDFABCDEGHI $.
  $}

  ${
    mp2and.1 $e |- ( ph -> ps ) $.
    mp2and.2 $e |- ( ph -> ch ) $.
    mp2and.3 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( A deduction based on modus ponens.  (Contributed by NM, 12-Dec-2004.) $)
    mp2and $p |- ( ph -> th ) $=
      ( mpand mpd ) ACDFABCDEGHI $.
  $}

  ${
    mpanl1.1 $e |- ph $.
    mpanl1.2 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 16-Aug-1994.)
       (Proof shortened by Wolf Lammen, 7-Apr-2013.) $)
    mpanl1 $p |- ( ( ps /\ ch ) -> th ) $=
      ( wa jctl sylan ) BABGCDBAEHFI $.
  $}

  ${
    mpanl2.1 $e |- ps $.
    mpanl2.2 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 16-Aug-1994.)
       (Proof shortened by Andrew Salmon, 7-May-2011.) $)
    mpanl2 $p |- ( ( ph /\ ch ) -> th ) $=
      ( wa jctr sylan ) AABGCDABEHFI $.
  $}

  ${
    mpanl12.1 $e |- ph $.
    mpanl12.2 $e |- ps $.
    mpanl12.3 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       13-Jul-2005.) $)
    mpanl12 $p |- ( ch -> th ) $=
      ( mpanl1 mpan ) BCDFABCDEGHI $.
  $}

  ${
    mpanr1.1 $e |- ps $.
    mpanr1.2 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 3-May-1994.)
       (Proof shortened by Andrew Salmon, 7-May-2011.) $)
    mpanr1 $p |- ( ( ph /\ ch ) -> th ) $=
      ( anassrs mpanl2 ) ABCDEABCDFGH $.
  $}

  ${
    mpanr2.1 $e |- ch $.
    mpanr2.2 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 3-May-1994.)
       (Proof shortened by Andrew Salmon, 7-May-2011.)  (Proof shortened by
       Wolf Lammen, 7-Apr-2013.) $)
    mpanr2 $p |- ( ( ph /\ ps ) -> th ) $=
      ( wa jctr sylan2 ) BABCGDBCEHFI $.
  $}

  ${
    mpanr12.1 $e |- ps $.
    mpanr12.2 $e |- ch $.
    mpanr12.3 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       24-Jul-2009.) $)
    mpanr12 $p |- ( ph -> th ) $=
      ( mpanr1 mpan2 ) ACDFABCDEGHI $.
  $}

  ${
    mpanlr1.1 $e |- ps $.
    mpanlr1.2 $e |- ( ( ( ph /\ ( ps /\ ch ) ) /\ th ) -> ta ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 30-Dec-2004.)
       (Proof shortened by Wolf Lammen, 7-Apr-2013.) $)
    mpanlr1 $p |- ( ( ( ph /\ ch ) /\ th ) -> ta ) $=
      ( wa jctl sylanl2 ) CABCHDECBFIGJ $.
  $}

  ${
    pm5.74da.1 $e |- ( ( ph /\ ps ) -> ( ch <-> th ) ) $.
    $( Distribution of implication over biconditional (deduction rule).
       (Contributed by NM, 4-May-2007.) $)
    pm5.74da $p |- ( ph -> ( ( ps -> ch ) <-> ( ps -> th ) ) ) $=
      ( wb ex pm5.74d ) ABCDABCDFEGH $.
  $}

  $( Theorem *4.45 of [WhiteheadRussell] p. 119.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.45 $p |- ( ph <-> ( ph /\ ( ph \/ ps ) ) ) $=
    ( wo orc pm4.71i ) AABCABDE $.

  $( Distribution of implication with conjunction.  (Contributed by NM,
     31-May-1999.)  (Proof shortened by Wolf Lammen, 6-Dec-2012.) $)
  imdistan $p |- ( ( ph -> ( ps -> ch ) ) <->
                ( ( ph /\ ps ) -> ( ph /\ ch ) ) ) $=
    ( wi wa pm5.42 impexp bitr4i ) ABCDDABACEZDDABEIDABCFABIGH $.

  ${
    imdistani.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Distribution of implication with conjunction.  (Contributed by NM,
       1-Aug-1994.) $)
    imdistani $p |- ( ( ph /\ ps ) -> ( ph /\ ch ) ) $=
      ( wa anc2li imp ) ABACEABCDFG $.
  $}

  ${
    imdistanri.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Distribution of implication with conjunction.  (Contributed by NM,
       8-Jan-2002.) $)
    imdistanri $p |- ( ( ps /\ ph ) -> ( ch /\ ph ) ) $=
      ( com12 impac ) BACABCDEF $.
  $}

  ${
    imdistand.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( Distribution of implication with conjunction (deduction rule).
       (Contributed by NM, 27-Aug-2004.) $)
    imdistand $p |- ( ph -> ( ( ps /\ ch ) -> ( ps /\ th ) ) ) $=
      ( wi wa imdistan sylib ) ABCDFFBCGBDGFEBCDHI $.
  $}

  ${
    imdistanda.1 $e |- ( ( ph /\ ps ) -> ( ch -> th ) ) $.
    $( Distribution of implication with conjunction (deduction version with
       conjoined antecedent).  (Contributed by Jeff Madsen, 19-Jun-2011.) $)
    imdistanda $p |- ( ph -> ( ( ps /\ ch ) -> ( ps /\ th ) ) ) $=
      ( wi ex imdistand ) ABCDABCDFEGH $.
  $}

  ${
    anbi.1 $e |- ( ph <-> ps ) $.
    $( Introduce a left conjunct to both sides of a logical equivalence.
       (Contributed by NM, 3-Jan-1993.)  (Proof shortened by Wolf Lammen,
       16-Nov-2013.) $)
    anbi2i $p |- ( ( ch /\ ph ) <-> ( ch /\ ps ) ) $=
      ( wb a1i pm5.32i ) CABABECDFG $.

    $( Introduce a right conjunct to both sides of a logical equivalence.
       (Contributed by NM, 12-Mar-1993.)  (Proof shortened by Wolf Lammen,
       16-Nov-2013.) $)
    anbi1i $p |- ( ( ph /\ ch ) <-> ( ps /\ ch ) ) $=
      ( wb a1i pm5.32ri ) CABABECDFG $.

    $( Variant of ~ anbi2i with commutation.  (Contributed by Jonathan
       Ben-Naim, 3-Jun-2011.)  (Proof shortened by Andrew Salmon,
       14-Jun-2011.) $)
    anbi2ci $p |- ( ( ph /\ ch ) <-> ( ch /\ ps ) ) $=
      ( wa anbi1i ancom bitri ) ACEBCECBEABCDFBCGH $.
  $}

  ${
    anbi12.1 $e |- ( ph <-> ps ) $.
    anbi12.2 $e |- ( ch <-> th ) $.
    $( Conjoin both sides of two equivalences.  (Contributed by NM,
       12-Mar-1993.) $)
    anbi12i $p |- ( ( ph /\ ch ) <-> ( ps /\ th ) ) $=
      ( wa anbi1i anbi2i bitri ) ACGBCGBDGABCEHCDBFIJ $.

    $( Variant of ~ anbi12i with commutation.  (Contributed by Jonathan
       Ben-Naim, 3-Jun-2011.) $)
    anbi12ci $p |- ( ( ph /\ ch ) <-> ( th /\ ps ) ) $=
      ( wa anbi12i ancom bitri ) ACGBDGDBGABCDEFHBDIJ $.
  $}

  ${
    sylan9bb.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    sylan9bb.2 $e |- ( th -> ( ch <-> ta ) ) $.
    $( Nested syllogism inference conjoining dissimilar antecedents.
       (Contributed by NM, 4-Mar-1995.) $)
    sylan9bb $p |- ( ( ph /\ th ) -> ( ps <-> ta ) ) $=
      ( wa wb adantr adantl bitrd ) ADHBCEABCIDFJDCEIAGKL $.
  $}

  ${
    sylan9bbr.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    sylan9bbr.2 $e |- ( th -> ( ch <-> ta ) ) $.
    $( Nested syllogism inference conjoining dissimilar antecedents.
       (Contributed by NM, 4-Mar-1995.) $)
    sylan9bbr $p |- ( ( th /\ ph ) -> ( ps <-> ta ) ) $=
      ( wb sylan9bb ancoms ) ADBEHABCDEFGIJ $.
  $}

  ${
    bid.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Deduction adding a left disjunct to both sides of a logical
       equivalence.  (Contributed by NM, 21-Jun-1993.) $)
    orbi2d $p |- ( ph -> ( ( th \/ ps ) <-> ( th \/ ch ) ) ) $=
      ( wn wi wo imbi2d df-or 3bitr4g ) ADFZBGLCGDBHDCHABCLEIDBJDCJK $.

    $( Deduction adding a right disjunct to both sides of a logical
       equivalence.  (Contributed by NM, 21-Jun-1993.) $)
    orbi1d $p |- ( ph -> ( ( ps \/ th ) <-> ( ch \/ th ) ) ) $=
      ( wo orbi2d orcom 3bitr4g ) ADBFDCFBDFCDFABCDEGBDHCDHI $.

    $( Deduction adding a left conjunct to both sides of a logical
       equivalence.  (Contributed by NM, 11-May-1993.)  (Proof shortened by
       Wolf Lammen, 16-Nov-2013.) $)
    anbi2d $p |- ( ph -> ( ( th /\ ps ) <-> ( th /\ ch ) ) ) $=
      ( wb a1d pm5.32d ) ADBCABCFDEGH $.

    $( Deduction adding a right conjunct to both sides of a logical
       equivalence.  (Contributed by NM, 11-May-1993.)  (Proof shortened by
       Wolf Lammen, 16-Nov-2013.) $)
    anbi1d $p |- ( ph -> ( ( ps /\ th ) <-> ( ch /\ th ) ) ) $=
      ( wb a1d pm5.32rd ) ADBCABCFDEGH $.
  $}

  $( Theorem *4.37 of [WhiteheadRussell] p. 118.  (Contributed by NM,
     3-Jan-2005.) $)
  orbi1 $p |- ( ( ph <-> ps ) -> ( ( ph \/ ch ) <-> ( ps \/ ch ) ) ) $=
    ( wb id orbi1d ) ABDZABCGEF $.

  $( Introduce a right conjunct to both sides of a logical equivalence.
     Theorem *4.36 of [WhiteheadRussell] p. 118.  (Contributed by NM,
     3-Jan-2005.) $)
  anbi1 $p |- ( ( ph <-> ps ) -> ( ( ph /\ ch ) <-> ( ps /\ ch ) ) ) $=
    ( wb id anbi1d ) ABDZABCGEF $.

  $( Introduce a left conjunct to both sides of a logical equivalence.
     (Contributed by NM, 16-Nov-2013.) $)
  anbi2 $p |- ( ( ph <-> ps ) -> ( ( ch /\ ph ) <-> ( ch /\ ps ) ) ) $=
    ( wb id anbi2d ) ABDZABCGEF $.

  $( Theorem *4.22 of [WhiteheadRussell] p. 117.  (Contributed by NM,
     3-Jan-2005.) $)
  bitr $p |- ( ( ( ph <-> ps ) /\ ( ps <-> ch ) ) -> ( ph <-> ch ) ) $=
    ( wb bibi1 biimpar ) ABDACDBCDABCEF $.

  ${
    bi12d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    bi12d.2 $e |- ( ph -> ( th <-> ta ) ) $.
    $( Deduction joining two equivalences to form equivalence of disjunctions.
       (Contributed by NM, 21-Jun-1993.) $)
    orbi12d $p |- ( ph -> ( ( ps \/ th ) <-> ( ch \/ ta ) ) ) $=
      ( wo orbi1d orbi2d bitrd ) ABDHCDHCEHABCDFIADECGJK $.

    $( Deduction joining two equivalences to form equivalence of conjunctions.
       (Contributed by NM, 26-May-1993.) $)
    anbi12d $p |- ( ph -> ( ( ps /\ th ) <-> ( ch /\ ta ) ) ) $=
      ( wa anbi1d anbi2d bitrd ) ABDHCDHCEHABCDFIADECGJK $.
  $}

  $( Theorem *5.3 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Andrew Salmon, 7-May-2011.) $)
  pm5.3 $p |- ( ( ( ph /\ ps ) -> ch ) <->
               ( ( ph /\ ps ) -> ( ph /\ ch ) ) ) $=
    ( wa wi impexp imdistan bitri ) ABDZCEABCEEIACDEABCFABCGH $.

  $( Theorem *5.61 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 30-Jun-2013.) $)
  pm5.61 $p |- ( ( ( ph \/ ps ) /\ -. ps ) <-> ( ph /\ -. ps ) ) $=
    ( wn wo biorf orcom syl6rbb pm5.32ri ) BCZABDZAIABADJBAEBAFGH $.

  ${
    adant2.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       4-May-1994.)  (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
    adantll $p |- ( ( ( th /\ ph ) /\ ps ) -> ch ) $=
      ( wa simpr sylan ) DAFABCDAGEH $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       4-May-1994.)  (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
    adantlr $p |- ( ( ( ph /\ th ) /\ ps ) -> ch ) $=
      ( wa simpl sylan ) ADFABCADGEH $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       4-May-1994.)  (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
    adantrl $p |- ( ( ph /\ ( th /\ ps ) ) -> ch ) $=
      ( wa simpr sylan2 ) DBFABCDBGEH $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       4-May-1994.)  (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
    adantrr $p |- ( ( ph /\ ( ps /\ th ) ) -> ch ) $=
      ( wa simpl sylan2 ) BDFABCBDGEH $.
  $}

  ${
    adantl2.1 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       26-Dec-2004.)  (Proof shortened by Wolf Lammen, 2-Dec-2012.) $)
    adantlll $p |- ( ( ( ( ta /\ ph ) /\ ps ) /\ ch ) -> th ) $=
      ( wa simpr sylanl1 ) EAGABCDEAHFI $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       26-Dec-2004.)  (Proof shortened by Wolf Lammen, 4-Dec-2012.) $)
    adantllr $p |- ( ( ( ( ph /\ ta ) /\ ps ) /\ ch ) -> th ) $=
      ( wa simpl sylanl1 ) AEGABCDAEHFI $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       26-Dec-2004.)  (Proof shortened by Wolf Lammen, 4-Dec-2012.) $)
    adantlrl $p |- ( ( ( ph /\ ( ta /\ ps ) ) /\ ch ) -> th ) $=
      ( wa simpr sylanl2 ) EBGABCDEBHFI $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       26-Dec-2004.)  (Proof shortened by Wolf Lammen, 4-Dec-2012.) $)
    adantlrr $p |- ( ( ( ph /\ ( ps /\ ta ) ) /\ ch ) -> th ) $=
      ( wa simpl sylanl2 ) BEGABCDBEHFI $.
  $}

  ${
    adantr2.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       26-Dec-2004.)  (Proof shortened by Wolf Lammen, 4-Dec-2012.) $)
    adantrll $p |- ( ( ph /\ ( ( ta /\ ps ) /\ ch ) ) -> th ) $=
      ( wa simpr sylanr1 ) EBGABCDEBHFI $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       26-Dec-2004.)  (Proof shortened by Wolf Lammen, 4-Dec-2012.) $)
    adantrlr $p |- ( ( ph /\ ( ( ps /\ ta ) /\ ch ) ) -> th ) $=
      ( wa simpl sylanr1 ) BEGABCDBEHFI $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       26-Dec-2004.)  (Proof shortened by Wolf Lammen, 4-Dec-2012.) $)
    adantrrl $p |- ( ( ph /\ ( ps /\ ( ta /\ ch ) ) ) -> th ) $=
      ( wa simpr sylanr2 ) ECGABCDECHFI $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       26-Dec-2004.)  (Proof shortened by Wolf Lammen, 4-Dec-2012.) $)
    adantrrr $p |- ( ( ph /\ ( ps /\ ( ch /\ ta ) ) ) -> th ) $=
      ( wa simpl sylanr2 ) CEGABCDCEHFI $.
  $}

  ${
    ad2ant.1 $e |- ( ph -> ps ) $.
    $( Deduction adding two conjuncts to antecedent.  (Contributed by NM,
       19-Oct-1999.)  (Proof shortened by Wolf Lammen, 20-Nov-2012.) $)
    ad2antrr $p |- ( ( ( ph /\ ch ) /\ th ) -> ps ) $=
      ( adantr adantlr ) ADBCABDEFG $.

    $( Deduction adding two conjuncts to antecedent.  (Contributed by NM,
       19-Oct-1999.)  (Proof shortened by Wolf Lammen, 20-Nov-2012.) $)
    ad2antlr $p |- ( ( ( ch /\ ph ) /\ th ) -> ps ) $=
      ( adantr adantll ) ADBCABDEFG $.

    $( Deduction adding two conjuncts to antecedent.  (Contributed by NM,
       19-Oct-1999.) $)
    ad2antrl $p |- ( ( ch /\ ( ph /\ th ) ) -> ps ) $=
      ( wa adantr adantl ) ADFBCABDEGH $.

    $( Deduction adding conjuncts to antecedent.  (Contributed by NM,
       19-Oct-1999.) $)
    ad2antll $p |- ( ( ch /\ ( th /\ ph ) ) -> ps ) $=
      ( wa adantl ) DAFBCABDEGG $.

    $( Deduction adding three conjuncts to antecedent.  (Contributed by NM,
       28-Jul-2012.) $)
    ad3antrrr $p |- ( ( ( ( ph /\ ch ) /\ th ) /\ ta ) -> ps ) $=
      ( wa adantr ad2antrr ) ACGBDEABCFHI $.

    $( Deduction adding three conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 5-Jan-2017.) $)
    ad3antlr $p |- ( ( ( ( ch /\ ph ) /\ th ) /\ ta ) -> ps ) $=
      ( wa ad2antlr adantr ) CAGDGBEABCDFHI $.

    $( Deduction adding 4 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 4-Jan-2017.) $)
    ad4antr $p |- ( ( ( ( ( ph /\ ch ) /\ th ) /\ ta ) /\ et ) -> ps ) $=
      ( wa ad3antrrr adantr ) ACHDHEHBFABCDEGIJ $.

    $( Deduction adding 4 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 5-Jan-2017.) $)
    ad4antlr $p |- ( ( ( ( ( ch /\ ph ) /\ th ) /\ ta ) /\ et ) -> ps ) $=
      ( wa ad3antlr adantr ) CAHDHEHBFABCDEGIJ $.

    $( Deduction adding 5 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 4-Jan-2017.) $)
    ad5antr $p |- ( ( ( ( ( ( ph /\ ch ) /\ th ) /\ ta ) /\ et )
      /\ ze ) -> ps ) $=
      ( wa ad4antr adantr ) ACIDIEIFIBGABCDEFHJK $.

    $( Deduction adding 5 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 5-Jan-2017.) $)
    ad5antlr $p |- ( ( ( ( ( ( ch /\ ph ) /\ th ) /\ ta ) /\ et )
      /\ ze ) -> ps ) $=
      ( wa ad4antlr adantr ) CAIDIEIFIBGABCDEFHJK $.

    $( Deduction adding 6 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 4-Jan-2017.) $)
    ad6antr $p |- ( ( ( ( ( ( ( ph /\ ch ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) -> ps ) $=
      ( wa ad5antr adantr ) ACJDJEJFJGJBHABCDEFGIKL $.

    $( Deduction adding 6 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 5-Jan-2017.) $)
    ad6antlr $p |- ( ( ( ( ( ( ( ch /\ ph ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) -> ps ) $=
      ( wa ad5antlr adantr ) CAJDJEJFJGJBHABCDEFGIKL $.

    $( Deduction adding 7 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 4-Jan-2017.) $)
    ad7antr $p |- ( ( ( ( ( ( ( ( ph /\ ch ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) /\ rh ) -> ps ) $=
      ( wa ad6antr adantr ) ACKDKEKFKGKHKBIABCDEFGHJLM $.

    $( Deduction adding 7 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 5-Jan-2017.) $)
    ad7antlr $p |- ( ( ( ( ( ( ( ( ch /\ ph ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) /\ rh ) -> ps ) $=
      ( wa ad6antlr adantr ) CAKDKEKFKGKHKBIABCDEFGHJLM $.

    $( Deduction adding 8 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 4-Jan-2017.) $)
    ad8antr $p |- ( ( ( ( ( ( ( ( ( ph /\ ch ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) /\ rh ) /\ mu ) -> ps ) $=
      ( wa ad7antr adantr ) ACLDLELFLGLHLILBJABCDEFGHIKMN $.

    $( Deduction adding 8 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 5-Jan-2017.) $)
    ad8antlr $p |- ( ( ( ( ( ( ( ( ( ch /\ ph ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) /\ rh ) /\ mu ) -> ps ) $=
      ( wa ad7antlr adantr ) CALDLELFLGLHLILBJABCDEFGHIKMN $.

    $( Deduction adding 9 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 4-Jan-2017.) $)
    ad9antr $p |- ( ( ( ( ( ( ( ( ( ( ph /\ ch ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la ) -> ps ) $=
      ( wa ad8antr adantr ) ACMDMEMFMGMHMIMJMBKABCDEFGHIJLNO $.

    $( Deduction adding 9 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 5-Jan-2017.) $)
    ad9antlr $p |- ( ( ( ( ( ( ( ( ( ( ch /\ ph ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la ) -> ps ) $=
      ( wa ad8antlr adantr ) CAMDMEMFMGMHMIMJMBKABCDEFGHIJLNO $.

    $( Deduction adding 10 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 4-Jan-2017.) $)
    ad10antr $p |- ( ( ( ( ( ( ( ( ( ( ( ph /\ ch ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la ) /\ ka ) -> ps ) $=
      ( wa ad9antr adantr ) ACNDNENFNGNHNINJNKNBLABCDEFGHIJKMOP $.

    $( Deduction adding 10 conjuncts to antecedent.  (Contributed by Mario
       Carneiro, 5-Jan-2017.) $)
    ad10antlr $p |- ( ( ( ( ( ( ( ( ( ( ( ch /\ ph ) /\ th ) /\ ta ) /\ et )
      /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la ) /\ ka ) -> ps ) $=
      ( wa ad9antlr adantr ) CANDNENFNGNHNINJNKNBLABCDEFGHIJKMOP $.
  $}

  ${
    ad2ant2.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( Deduction adding two conjuncts to antecedent.  (Contributed by NM,
       8-Jan-2006.) $)
    ad2ant2l $p |- ( ( ( th /\ ph ) /\ ( ta /\ ps ) ) -> ch ) $=
      ( wa adantrl adantll ) AEBGCDABCEFHI $.

    $( Deduction adding two conjuncts to antecedent.  (Contributed by NM,
       8-Jan-2006.) $)
    ad2ant2r $p |- ( ( ( ph /\ th ) /\ ( ps /\ ta ) ) -> ch ) $=
      ( wa adantrr adantlr ) ABEGCDABCEFHI $.

    $( Deduction adding two conjuncts to antecedent.  (Contributed by NM,
       23-Nov-2007.) $)
    ad2ant2lr $p |- ( ( ( th /\ ph ) /\ ( ps /\ ta ) ) -> ch ) $=
      ( wa adantrr adantll ) ABEGCDABCEFHI $.

    $( Deduction adding two conjuncts to antecedent.  (Contributed by NM,
       24-Nov-2007.) $)
    ad2ant2rl $p |- ( ( ( ph /\ th ) /\ ( ta /\ ps ) ) -> ch ) $=
      ( wa adantrl adantlr ) AEBGCDABCEFHI $.
  $}

  ${
    adantl3r.1 $e |- ( ( ( ( ph /\ rh ) /\ mu ) /\ la ) -> ka ) $.
    $( Deduction adding 1 conjunct to antecedent.  (Contributed by Thierry
       Arnoux, 11-Feb-2018.) $)
    adantl3r $p |- ( ( ( ( ( ph /\ si ) /\ rh ) /\ mu ) /\ la ) -> ka ) $=
      ( wa wi ex adantllr imp ) ABHCHDHEFACDEFIBACHDHEFGJKL $.
  $}

  ${
    adantl4r.1 $e |- ( ( ( ( ( ph /\ si ) /\ rh ) /\ mu ) /\ la ) -> ka ) $.
    $( Deduction adding 1 conjunct to antecedent.  (Contributed by Thierry
       Arnoux, 11-Feb-2018.) $)
    adantl4r $p |- ( ( ( ( ( ( ph /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la )
      -> ka ) $=
      ( wa wi ex adantl3r imp ) ABICIDIEIFGABCDEFGJACIDIEIFGHKLM $.
  $}

  ${
    adantl5r.1 $e |- ( ( ( ( ( ( ph /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la )
      -> ka ) $.
    $( Deduction adding 1 conjunct to antecedent.  (Contributed by Thierry
       Arnoux, 11-Feb-2018.) $)
    adantl5r $p |- ( ( ( ( ( ( ( ph /\ et ) /\ ze ) /\ si ) /\ rh ) /\ mu )
      /\ la ) -> ka ) $=
      ( wa wi ex adantl4r imp ) ABJCJDJEJFJGHABCDEFGHKACJDJEJFJGHILMN $.
  $}

  ${
    adantl6r.1 $e |- ( ( ( ( ( ( ( ph /\ et ) /\ ze ) /\ si ) /\ rh ) /\ mu )
      /\ la ) -> ka ) $.
    $( Deduction adding 1 conjunct to antecedent.  (Contributed by Thierry
       Arnoux, 11-Feb-2018.) $)
    adantl6r $p |- ( ( ( ( ( ( ( ( ph /\ ta ) /\ et ) /\ ze ) /\ si ) /\ rh )
      /\ mu ) /\ la ) -> ka ) $=
      ( wa wi ex adantl5r imp ) ABKCKDKEKFKGKHIABCDEFGHILACKDKEKFKGKHIJMNO $.
  $}

  $( Simplification of a conjunction.  (Contributed by NM, 18-Mar-2007.) $)
  simpll $p |- ( ( ( ph /\ ps ) /\ ch ) -> ph ) $=
    ( id ad2antrr ) AABCADE $.

  ${
    simplld.1 $e |- ( ph -> ( ( ps /\ ch ) /\ th ) ) $.
    $( Deduction form of ~ simpll , eliminating a double conjunct.
       (Contributed by Glauco Siliprandi, 11-Dec-2019.) $)
    simplld $p |- ( ph -> ps ) $=
      ( wa simpld ) ABCABCFDEGG $.
  $}

  $( Simplification of a conjunction.  (Contributed by NM, 20-Mar-2007.) $)
  simplr $p |- ( ( ( ph /\ ps ) /\ ch ) -> ps ) $=
    ( id ad2antlr ) BBACBDE $.

  $( Simplification of a conjunction.  (Contributed by NM, 21-Mar-2007.) $)
  simprl $p |- ( ( ph /\ ( ps /\ ch ) ) -> ps ) $=
    ( id ad2antrl ) BBACBDE $.

  $( Simplification of a conjunction.  (Contributed by NM, 21-Mar-2007.) $)
  simprr $p |- ( ( ph /\ ( ps /\ ch ) ) -> ch ) $=
    ( id ad2antll ) CCABCDE $.

  ${
    simprrd.1 $e |- ( ph -> ( ps /\ ( ch /\ th ) ) ) $.
    $( Deduction form of ~ simprr , eliminating a double conjunct.
       (Contributed by Glauco Siliprandi, 11-Dec-2019.) $)
    simprrd $p |- ( ph -> th ) $=
      ( wa simprd ) ACDABCDFEGG $.
  $}

  $( Simplification of a conjunction.  (Contributed by Jeff Hankins,
     28-Jul-2009.) $)
  simplll $p |- ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) -> ph ) $=
    ( wa simpl ad2antrr ) ABEACDABFG $.

  $( Simplification of a conjunction.  (Contributed by Jeff Hankins,
     28-Jul-2009.) $)
  simpllr $p |- ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) -> ps ) $=
    ( wa simpr ad2antrr ) ABEBCDABFG $.

  $( Simplification of a conjunction.  (Contributed by Jeff Hankins,
     28-Jul-2009.) $)
  simplrl $p |- ( ( ( ph /\ ( ps /\ ch ) ) /\ th ) -> ps ) $=
    ( wa simpl ad2antlr ) BCEBADBCFG $.

  $( Simplification of a conjunction.  (Contributed by Jeff Hankins,
     28-Jul-2009.) $)
  simplrr $p |- ( ( ( ph /\ ( ps /\ ch ) ) /\ th ) -> ch ) $=
    ( wa simpr ad2antlr ) BCECADBCFG $.

  $( Simplification of a conjunction.  (Contributed by Jeff Hankins,
     28-Jul-2009.) $)
  simprll $p |- ( ( ph /\ ( ( ps /\ ch ) /\ th ) ) -> ps ) $=
    ( wa simpl ad2antrl ) BCEBADBCFG $.

  $( Simplification of a conjunction.  (Contributed by Jeff Hankins,
     28-Jul-2009.) $)
  simprlr $p |- ( ( ph /\ ( ( ps /\ ch ) /\ th ) ) -> ch ) $=
    ( wa simpr ad2antrl ) BCECADBCFG $.

  $( Simplification of a conjunction.  (Contributed by Jeff Hankins,
     28-Jul-2009.) $)
  simprrl $p |- ( ( ph /\ ( ps /\ ( ch /\ th ) ) ) -> ch ) $=
    ( wa simpl ad2antll ) CDECABCDFG $.

  $( Simplification of a conjunction.  (Contributed by Jeff Hankins,
     28-Jul-2009.) $)
  simprrr $p |- ( ( ph /\ ( ps /\ ( ch /\ th ) ) ) -> th ) $=
    ( wa simpr ad2antll ) CDEDABCDFG $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-4l $p |- ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta ) -> ph ) $=
    ( wa simplll adantr ) ABFCFDFAEABCDGH $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-4r $p |- ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta ) -> ps ) $=
    ( wa simpllr adantr ) ABFCFDFBEABCDGH $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-5l $p |- ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) -> ph ) $=
    ( wa simp-4l adantr ) ABGCGDGEGAFABCDEHI $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-5r $p |- ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) -> ps ) $=
    ( wa simp-4r adantr ) ABGCGDGEGBFABCDEHI $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-6l $p |- ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) -> ph ) $=
    ( wa simp-5l adantr ) ABHCHDHEHFHAGABCDEFIJ $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-6r $p |- ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) -> ps ) $=
    ( wa simp-5r adantr ) ABHCHDHEHFHBGABCDEFIJ $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-7l $p |- ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) -> ph ) $=
    ( wa simp-6l adantr ) ABICIDIEIFIGIAHABCDEFGJK $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-7r $p |- ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) -> ps ) $=
    ( wa simp-6r adantr ) ABICIDIEIFIGIBHABCDEFGJK $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-8l $p |- ( ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) /\ rh ) -> ph ) $=
    ( wa simp-7l adantr ) ABJCJDJEJFJGJHJAIABCDEFGHKL $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-8r $p |- ( ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) /\ rh ) -> ps ) $=
    ( wa simp-7r adantr ) ABJCJDJEJFJGJHJBIABCDEFGHKL $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-9l $p |- ( ( ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) /\ rh ) /\ mu ) -> ph ) $=
    ( wa simp-8l adantr ) ABKCKDKEKFKGKHKIKAJABCDEFGHILM $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-9r $p |- ( ( ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) /\ rh ) /\ mu ) -> ps ) $=
    ( wa simp-8r adantr ) ABKCKDKEKFKGKHKIKBJABCDEFGHILM $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-10l $p |- ( ( ( ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la ) -> ph ) $=
    ( wa simp-9l adantr ) ABLCLDLELFLGLHLILJLAKABCDEFGHIJMN $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-10r $p |- ( ( ( ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la ) -> ps ) $=
    ( wa simp-9r adantr ) ABLCLDLELFLGLHLILJLBKABCDEFGHIJMN $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-11l $p |- ( ( ( ( ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la ) /\ ka ) -> ph ) $=
    ( wa simp-10l adantr ) ABMCMDMEMFMGMHMIMJMKMALABCDEFGHIJKNO $.

  $( Simplification of a conjunction.  (Contributed by Mario Carneiro,
     4-Jan-2017.) $)
  simp-11r $p |- ( ( ( ( ( ( ( ( ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) /\ ta )
    /\ et ) /\ ze ) /\ si ) /\ rh ) /\ mu ) /\ la ) /\ ka ) -> ps ) $=
    ( wa simp-10r adantr ) ABMCMDMEMFMGMHMIMJMKMBLABCDEFGHIJKNO $.

  $( Disjunction of antecedents.  Compare Theorem *4.77 of [WhiteheadRussell]
     p. 121.  (Contributed by NM, 30-May-1994.)  (Proof shortened by Wolf
     Lammen, 9-Dec-2012.) $)
  jaob $p |- ( ( ( ph \/ ch ) -> ps ) <-> ( ( ph -> ps ) /\ ( ch -> ps ) ) ) $=
    ( wo wi wa pm2.67-2 olc imim1i jca pm3.44 impbii ) ACDZBEZABEZCBEZFNOPABCGC
    MBCAHIJBACKL $.

  ${
    jaoian.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    jaoian.2 $e |- ( ( th /\ ps ) -> ch ) $.
    $( Inference disjoining the antecedents of two implications.  (Contributed
       by NM, 23-Oct-2005.) $)
    jaoian $p |- ( ( ( ph \/ th ) /\ ps ) -> ch ) $=
      ( wo wi ex jaoi imp ) ADGBCABCHDABCEIDBCFIJK $.
  $}

  ${
    jaodan.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    jaodan.2 $e |- ( ( ph /\ th ) -> ch ) $.
    $( Deduction disjoining the antecedents of two implications.  (Contributed
       by NM, 14-Oct-2005.) $)
    jaodan $p |- ( ( ph /\ ( ps \/ th ) ) -> ch ) $=
      ( wo ex jaod imp ) ABDGCABCDABCEHADCFHIJ $.

    jaodan.3 $e |- ( ph -> ( ps \/ th ) ) $.
    $( Eliminate a disjunction in a deduction.  A translation of natural
       deduction rule ` \/ ` E ( ` \/ ` elimination), see ~ natded .
       (Contributed by Mario Carneiro, 29-May-2016.) $)
    mpjaodan $p |- ( ph -> ch ) $=
      ( wo jaodan mpdan ) ABDHCGABCDEFIJ $.
  $}

  $( Theorem *4.77 of [WhiteheadRussell] p. 121.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.77 $p |- ( ( ( ps -> ph ) /\ ( ch -> ph ) ) <->
                ( ( ps \/ ch ) -> ph ) ) $=
    ( wo wi wa jaob bicomi ) BCDAEBAECAEFBACGH $.

  $( Theorem *2.63 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.63 $p |- ( ( ph \/ ps ) -> ( ( -. ph \/ ps ) -> ps ) ) $=
    ( wo wn pm2.53 idd jaod ) ABCZADBBABEHBFG $.

  $( Theorem *2.64 of [WhiteheadRussell] p. 107.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.64 $p |- ( ( ph \/ ps ) -> ( ( ph \/ -. ps ) -> ph ) ) $=
    ( wn wo wi ax-1 orel2 jaoi com12 ) ABCZDABDZAAKAEJAKFBAGHI $.

  ${
    pm2.61ian.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    pm2.61ian.2 $e |- ( ( -. ph /\ ps ) -> ch ) $.
    $( Elimination of an antecedent.  (Contributed by NM, 1-Jan-2005.) $)
    pm2.61ian $p |- ( ps -> ch ) $=
      ( wi ex wn pm2.61i ) ABCFABCDGAHBCEGI $.
  $}

  ${
    pm2.61dan.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    pm2.61dan.2 $e |- ( ( ph /\ -. ps ) -> ch ) $.
    $( Elimination of an antecedent.  (Contributed by NM, 1-Jan-2005.) $)
    pm2.61dan $p |- ( ph -> ch ) $=
      ( ex wn pm2.61d ) ABCABCDFABGCEFH $.
  $}

  ${
    pm2.61ddan.1 $e |- ( ( ph /\ ps ) -> th ) $.
    pm2.61ddan.2 $e |- ( ( ph /\ ch ) -> th ) $.
    pm2.61ddan.3 $e |- ( ( ph /\ ( -. ps /\ -. ch ) ) -> th ) $.
    $( Elimination of two antecedents.  (Contributed by NM, 9-Jul-2013.) $)
    pm2.61ddan $p |- ( ph -> th ) $=
      ( wn wa adantlr anassrs pm2.61dan ) ABDEABHZICDACDMFJAMCHDGKLL $.
  $}

  ${
    pm2.61dda.1 $e |- ( ( ph /\ -. ps ) -> th ) $.
    pm2.61dda.2 $e |- ( ( ph /\ -. ch ) -> th ) $.
    pm2.61dda.3 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( Elimination of two antecedents.  (Contributed by NM, 9-Jul-2013.) $)
    pm2.61dda $p |- ( ph -> th ) $=
      ( wa anassrs wn adantlr pm2.61dan ) ABDABHCDABCDGIACJDBFKLEL $.
  $}

  ${
    condan.1 $e |- ( ( ph /\ -. ps ) -> ch ) $.
    condan.2 $e |- ( ( ph /\ -. ps ) -> -. ch ) $.
    $( Proof by contradiction.  (Contributed by NM, 9-Feb-2006.)  (Proof
       shortened by Wolf Lammen, 19-Jun-2014.) $)
    condan $p |- ( ph -> ps ) $=
      ( wn pm2.65da notnotrd ) ABABFCDEGH $.
  $}

  $( Introduce one conjunct as an antecedent to the other.  "abai" stands for
     "and, biconditional, and, implication".  (Contributed by NM,
     12-Aug-1993.)  (Proof shortened by Wolf Lammen, 7-Dec-2012.) $)
  abai $p |- ( ( ph /\ ps ) <-> ( ph /\ ( ph -> ps ) ) ) $=
    ( wi biimt pm5.32i ) ABABCABDE $.

  $( Theorem *5.53 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.53 $p |- ( ( ( ( ph \/ ps ) \/ ch ) -> th ) <->
                ( ( ( ph -> th ) /\ ( ps -> th ) ) /\ ( ch -> th ) ) ) $=
    ( wo wi wa jaob anbi1i bitri ) ABEZCEDFKDFZCDFZGADFBDFGZMGKDCHLNMADBHIJ $.

  $( Swap two conjuncts.  Note that the first digit (1) in the label refers to
     the outer conjunct position, and the next digit (2) to the inner conjunct
     position.  (Contributed by NM, 12-Mar-1995.) $)
  an12 $p |- ( ( ph /\ ( ps /\ ch ) ) <-> ( ps /\ ( ph /\ ch ) ) ) $=
    ( wa ancom anbi1i anass 3bitr3i ) ABDZCDBADZCDABCDDBACDDIJCABEFABCGBACGH $.

  $( A rearrangement of conjuncts.  (Contributed by NM, 12-Mar-1995.)  (Proof
     shortened by Wolf Lammen, 25-Dec-2012.) $)
  an32 $p |- ( ( ( ph /\ ps ) /\ ch ) <-> ( ( ph /\ ch ) /\ ps ) ) $=
    ( wa anass an12 ancom 3bitri ) ABDCDABCDDBACDZDIBDABCEABCFBIGH $.

  $( A rearrangement of conjuncts.  (Contributed by NM, 24-Jun-2012.)  (Proof
     shortened by Wolf Lammen, 31-Dec-2012.) $)
  an13 $p |- ( ( ph /\ ( ps /\ ch ) ) <-> ( ch /\ ( ps /\ ph ) ) ) $=
    ( wa an12 anass ancom 3bitr2i ) ABCDDBACDDBADZCDCIDABCEBACFICGH $.

  $( A rearrangement of conjuncts.  (Contributed by NM, 24-Jun-2012.)  (Proof
     shortened by Wolf Lammen, 31-Dec-2012.) $)
  an31 $p |- ( ( ( ph /\ ps ) /\ ch ) <-> ( ( ch /\ ps ) /\ ph ) ) $=
    ( wa an13 anass 3bitr4i ) ABCDDCBADDABDCDCBDADABCEABCFCBAFG $.

  ${
    an12s.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( Swap two conjuncts in antecedent.  The label suffix "s" means that
       ~ an12 is combined with ~ syl (or a variant).  (Contributed by NM,
       13-Mar-1996.) $)
    an12s $p |- ( ( ps /\ ( ph /\ ch ) ) -> th ) $=
      ( wa an12 sylbi ) BACFFABCFFDBACGEH $.

    $( Inference commuting a nested conjunction in antecedent.  (Contributed by
       NM, 24-May-2006.)  (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
    ancom2s $p |- ( ( ph /\ ( ch /\ ps ) ) -> th ) $=
      ( wa pm3.22 sylan2 ) CBFABCFDCBGEH $.

    $( Swap two conjuncts in antecedent.  (Contributed by NM, 31-May-2006.) $)
    an13s $p |- ( ( ch /\ ( ps /\ ph ) ) -> th ) $=
      ( exp32 com13 imp32 ) CBADABCDABCDEFGH $.
  $}

  ${
    an32s.1 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( Swap two conjuncts in antecedent.  (Contributed by NM, 13-Mar-1996.) $)
    an32s $p |- ( ( ( ph /\ ch ) /\ ps ) -> th ) $=
      ( wa an32 sylbi ) ACFBFABFCFDACBGEH $.

    $( Inference commuting a nested conjunction in antecedent.  (Contributed by
       NM, 24-May-2006.)  (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
    ancom1s $p |- ( ( ( ps /\ ph ) /\ ch ) -> th ) $=
      ( wa pm3.22 sylan ) BAFABFCDBAGEH $.

    $( Swap two conjuncts in antecedent.  (Contributed by NM, 31-May-2006.) $)
    an31s $p |- ( ( ( ch /\ ps ) /\ ph ) -> th ) $=
      ( exp31 com13 imp31 ) CBADABCDABCDEFGH $.
  $}

  ${
    anass1rs.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( Commutative-associative law for conjunction in an antecedent.
       (Contributed by Jeff Madsen, 19-Jun-2011.) $)
    anass1rs $p |- ( ( ( ph /\ ch ) /\ ps ) -> th ) $=
      ( anassrs an32s ) ABCDABCDEFG $.
  $}

  $( Absorption into embedded conjunct.  (Contributed by NM, 4-Sep-1995.)
     (Proof shortened by Wolf Lammen, 16-Nov-2013.) $)
  anabs1 $p |- ( ( ( ph /\ ps ) /\ ph ) <-> ( ph /\ ps ) ) $=
    ( wa simpl pm4.71i bicomi ) ABCZGACGAABDEF $.

  $( Absorption into embedded conjunct.  (Contributed by NM, 20-Jul-1996.)
     (Proof shortened by Wolf Lammen, 9-Dec-2012.) $)
  anabs5 $p |- ( ( ph /\ ( ph /\ ps ) ) <-> ( ph /\ ps ) ) $=
    ( wa ibar bicomd pm5.32i ) AABCZBABGABDEF $.

  $( Absorption into embedded conjunct.  (Contributed by NM, 20-Jul-1996.)
     (Proof shortened by Wolf Lammen, 17-Nov-2013.) $)
  anabs7 $p |- ( ( ps /\ ( ph /\ ps ) ) <-> ( ph /\ ps ) ) $=
    ( wa simpr pm4.71ri bicomi ) ABCZBGCGBABDEF $.

  ${
    a2and.1 $e |- ( ph -> ( ( ps /\ rh ) -> ( ta -> th ) ) ) $.
    a2and.2 $e |- ( ph -> ( ( ps /\ rh ) -> ch ) ) $.
    $( Deduction distributing a conjunction as embedded antecedent.
       (Contributed by AV, 25-Oct-2019.)  (Proof shortened by Wolf Lammen,
       19-Jan-2020.) $)
    a2and $p |- ( ph -> ( ( ( ps /\ ch ) -> ta )
                            -> ( ( ps /\ rh ) -> th ) ) ) $=
      ( wa wi expd imdistand imp embantd ex com23 ) ABFIZBCIZEJZDAQSDJAQIREDAQR
      ABFCABFCHKLMAQEDJGMNOP $.
    $( Obsolete proof of ~ a2and as of 19-Jan-2020.  (Contributed by AV,
       25-Oct-2019.)  (Proof modification is discouraged.)
       (New usage is discouraged.) $)
    a2andOLD $p |- ( ph -> ( ( ( ps /\ ch ) -> ta )
                            -> ( ( ps /\ rh ) -> th ) ) ) $=
      ( wa wi pm2.27 adantlr syl6 ex syld com24 com14 sylc ) ABFIZCJZSEDJZJZBCI
      ZEJZSDJJHGSUBUDTDSUBUAUDTDJJSUAKSTUDUADSTCUDUADJZJZSCKSCUFSCIUDEUEBCUDEJF
      UCEKLEDKMNOPOQR $.
  $}

  ${
    anabsan.1 $e |- ( ( ( ph /\ ph ) /\ ps ) -> ch ) $.
    $( Absorption of antecedent with conjunction.  (Contributed by NM,
       24-Mar-1996.) $)
    anabsan $p |- ( ( ph /\ ps ) -> ch ) $=
      ( wa pm4.24 sylanb ) AAAEBCAFDG $.
  $}

  ${
    anabss1.1 $e |- ( ( ( ph /\ ps ) /\ ph ) -> ch ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       20-Jul-1996.)  (Proof shortened by Wolf Lammen, 31-Dec-2012.) $)
    anabss1 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( an32s anabsan ) ABCABACDEF $.
  $}

  ${
    anabss4.1 $e |- ( ( ( ps /\ ph ) /\ ps ) -> ch ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       20-Jul-1996.) $)
    anabss4 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( anabss1 ancoms ) BACBACDEF $.
  $}

  ${
    anabss5.1 $e |- ( ( ph /\ ( ph /\ ps ) ) -> ch ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       10-May-1994.)  (Proof shortened by Wolf Lammen, 1-Jan-2013.) $)
    anabss5 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( anassrs anabsan ) ABCAABCDEF $.
  $}

  ${
    anabsi5.1 $e |- ( ph -> ( ( ph /\ ps ) -> ch ) ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       11-Jun-1995.)  (Proof shortened by Wolf Lammen, 18-Nov-2013.) $)
    anabsi5 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( wa imp anabss5 ) ABCAABECDFG $.
  $}

  ${
    anabsi6.1 $e |- ( ph -> ( ( ps /\ ph ) -> ch ) ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       14-Aug-2000.) $)
    anabsi6 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( ancomsd anabsi5 ) ABCABACDEF $.
  $}

  ${
    anabsi7.1 $e |- ( ps -> ( ( ph /\ ps ) -> ch ) ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       20-Jul-1996.)  (Proof shortened by Wolf Lammen, 18-Nov-2013.) $)
    anabsi7 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( anabsi6 ancoms ) BACBACDEF $.
  $}

  ${
    anabsi8.1 $e |- ( ps -> ( ( ps /\ ph ) -> ch ) ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       26-Sep-1999.) $)
    anabsi8 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( anabsi5 ancoms ) BACBACDEF $.
  $}

  ${
    anabss7.1 $e |- ( ( ps /\ ( ph /\ ps ) ) -> ch ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       20-Jul-1996.)  (Proof shortened by Wolf Lammen, 19-Nov-2013.) $)
    anabss7 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( anassrs anabss4 ) ABCBABCDEF $.
  $}

  ${
    anabsan2.1 $e |- ( ( ph /\ ( ps /\ ps ) ) -> ch ) $.
    $( Absorption of antecedent with conjunction.  (Contributed by NM,
       10-May-2004.) $)
    anabsan2 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( an12s anabss7 ) ABCABBCDEF $.
  $}

  ${
    anabss3.1 $e |- ( ( ( ph /\ ps ) /\ ps ) -> ch ) $.
    $( Absorption of antecedent into conjunction.  (Contributed by NM,
       20-Jul-1996.)  (Proof shortened by Wolf Lammen, 1-Jan-2013.) $)
    anabss3 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( anasss anabsan2 ) ABCABBCDEF $.
  $}

  $( Rearrangement of 4 conjuncts.  (Contributed by NM, 10-Jul-1994.) $)
  an4 $p |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) <->
              ( ( ph /\ ch ) /\ ( ps /\ th ) ) ) $=
    ( wa an12 anbi2i anass 3bitr4i ) ABCDEZEZEACBDEZEZEABEJEACELEKMABCDFGABJHAC
    LHI $.

  $( Rearrangement of 4 conjuncts.  (Contributed by NM, 7-Feb-1996.) $)
  an42 $p |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) <->
                 ( ( ph /\ ch ) /\ ( th /\ ps ) ) ) $=
    ( wa an4 ancom anbi2i bitri ) ABECDEEACEZBDEZEJDBEZEABCDFKLJBDGHI $.

  ${
    an4s.1 $e |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) -> ta ) $.
    $( Inference rearranging 4 conjuncts in antecedent.  (Contributed by NM,
       10-Aug-1995.) $)
    an4s $p |- ( ( ( ph /\ ch ) /\ ( ps /\ th ) ) -> ta ) $=
      ( wa an4 sylbi ) ACGBDGGABGCDGGEACBDHFI $.
  $}

  ${
    an41r3s.1 $e |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) -> ta ) $.
    $( Inference rearranging 4 conjuncts in antecedent.  (Contributed by NM,
       10-Aug-1995.) $)
    an42s $p |- ( ( ( ph /\ ch ) /\ ( th /\ ps ) ) -> ta ) $=
      ( wa an4s ancom2s ) ACGBDEABCDEFHI $.
  $}

  $( Distribution of conjunction over conjunction.  (Contributed by NM,
     14-Aug-1995.) $)
  anandi $p |- ( ( ph /\ ( ps /\ ch ) ) <->
               ( ( ph /\ ps ) /\ ( ph /\ ch ) ) ) $=
    ( wa anidm anbi1i an4 bitr3i ) ABCDZDAADZIDABDACDDJAIAEFAABCGH $.

  $( Distribution of conjunction over conjunction.  (Contributed by NM,
     24-Aug-1995.) $)
  anandir $p |- ( ( ( ph /\ ps ) /\ ch ) <->
               ( ( ph /\ ch ) /\ ( ps /\ ch ) ) ) $=
    ( wa anidm anbi2i an4 bitr3i ) ABDZCDICCDZDACDBCDDJCICEFABCCGH $.

  ${
    anandis.1 $e |- ( ( ( ph /\ ps ) /\ ( ph /\ ch ) ) -> ta ) $.
    $( Inference that undistributes conjunction in the antecedent.
       (Contributed by NM, 7-Jun-2004.) $)
    anandis $p |- ( ( ph /\ ( ps /\ ch ) ) -> ta ) $=
      ( wa an4s anabsan ) ABCFDABACDEGH $.
  $}

  ${
    anandirs.1 $e |- ( ( ( ph /\ ch ) /\ ( ps /\ ch ) ) -> ta ) $.
    $( Inference that undistributes conjunction in the antecedent.
       (Contributed by NM, 7-Jun-2004.) $)
    anandirs $p |- ( ( ( ph /\ ps ) /\ ch ) -> ta ) $=
      ( wa an4s anabsan2 ) ABFCDACBCDEGH $.
  $}

  ${
    impbida.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    impbida.2 $e |- ( ( ph /\ ch ) -> ps ) $.
    $( Deduce an equivalence from two implications.  (Contributed by NM,
       17-Feb-2007.) $)
    impbida $p |- ( ph -> ( ps <-> ch ) ) $=
      ( ex impbid ) ABCABCDFACBEFG $.
  $}

  $( Theorem *3.48 of [WhiteheadRussell] p. 114.  (Contributed by NM,
     28-Jan-1997.) $)
  pm3.48 $p |- ( ( ( ph -> ps ) /\ ( ch -> th ) )
      -> ( ( ph \/ ch ) -> ( ps \/ th ) ) ) $=
    ( wi wo orc imim2i olc jaao ) ABEABDFZCDECBKABDGHDKCDBIHJ $.

  $( Theorem *3.45 (Fact) of [WhiteheadRussell] p. 113.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.45 $p |- ( ( ph -> ps ) -> ( ( ph /\ ch ) -> ( ps /\ ch ) ) ) $=
    ( wi id anim1d ) ABDZABCGEF $.

  ${
    im2an9.1 $e |- ( ph -> ( ps -> ch ) ) $.
    im2an9.2 $e |- ( th -> ( ta -> et ) ) $.
    $( Deduction joining nested implications to form implication of
       conjunctions.  (Contributed by NM, 29-Feb-1996.) $)
    im2anan9 $p |- ( ( ph /\ th ) -> ( ( ps /\ ta ) -> ( ch /\ et ) ) ) $=
      ( wa wi adantr adantl anim12d ) ADIBCEFABCJDGKDEFJAHLM $.

    $( Deduction joining nested implications to form implication of
       conjunctions.  (Contributed by NM, 29-Feb-1996.) $)
    im2anan9r $p |- ( ( th /\ ph ) -> ( ( ps /\ ta ) -> ( ch /\ et ) ) ) $=
      ( wa wi im2anan9 ancoms ) ADBEICFIJABCDEFGHKL $.
  $}

  ${
    anim12dan.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    anim12dan.2 $e |- ( ( ph /\ th ) -> ta ) $.
    $( Conjoin antecedents and consequents in a deduction.  (Contributed by
       Mario Carneiro, 12-May-2014.) $)
    anim12dan $p |- ( ( ph /\ ( ps /\ th ) ) -> ( ch /\ ta ) ) $=
      ( wa ex anim12d imp ) ABDHCEHABCDEABCFIADEGIJK $.
  $}

  ${
    orim12d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    orim12d.2 $e |- ( ph -> ( th -> ta ) ) $.
    $( Disjoin antecedents and consequents in a deduction.  (Contributed by NM,
       10-May-1994.) $)
    orim12d $p |- ( ph -> ( ( ps \/ th ) -> ( ch \/ ta ) ) ) $=
      ( wi wo pm3.48 syl2anc ) ABCHDEHBDICEIHFGBCDEJK $.
  $}

  ${
    orim1d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    $( Disjoin antecedents and consequents in a deduction.  (Contributed by NM,
       23-Apr-1995.) $)
    orim1d $p |- ( ph -> ( ( ps \/ th ) -> ( ch \/ th ) ) ) $=
      ( idd orim12d ) ABCDDEADFG $.

    $( Disjoin antecedents and consequents in a deduction.  (Contributed by NM,
       23-Apr-1995.) $)
    orim2d $p |- ( ph -> ( ( th \/ ps ) -> ( th \/ ch ) ) ) $=
      ( idd orim12d ) ADDBCADFEG $.
  $}

  $( Axiom *1.6 (Sum) of [WhiteheadRussell] p. 97.  (Contributed by NM,
     3-Jan-2005.) $)
  orim2 $p |- ( ( ps -> ch ) -> ( ( ph \/ ps ) -> ( ph \/ ch ) ) ) $=
    ( wi id orim2d ) BCDZBCAGEF $.

  $( Theorem *2.38 of [WhiteheadRussell] p. 105.  (Contributed by NM,
     6-Mar-2008.) $)
  pm2.38 $p |- ( ( ps -> ch ) -> ( ( ps \/ ph ) -> ( ch \/ ph ) ) ) $=
    ( wi id orim1d ) BCDZBCAGEF $.

  $( Theorem *2.36 of [WhiteheadRussell] p. 105.  (Contributed by NM,
     6-Mar-2008.) $)
  pm2.36 $p |- ( ( ps -> ch ) -> ( ( ph \/ ps ) -> ( ch \/ ph ) ) ) $=
    ( wo wi pm1.4 pm2.38 syl5 ) ABDBADBCECADABFABCGH $.

  $( Theorem *2.37 of [WhiteheadRussell] p. 105.  (Contributed by NM,
     6-Mar-2008.) $)
  pm2.37 $p |- ( ( ps -> ch ) -> ( ( ps \/ ph ) -> ( ph \/ ch ) ) ) $=
    ( wi wo pm2.38 pm1.4 syl6 ) BCDBAECAEACEABCFCAGH $.

  $( Theorem *2.73 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.73 $p |- ( ( ph -> ps )
       -> ( ( ( ph \/ ps ) \/ ch ) -> ( ps \/ ch ) ) ) $=
    ( wi wo pm2.621 orim1d ) ABDABEBCABFG $.

  $( Theorem *2.74 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Andrew Salmon, 7-May-2011.) $)
  pm2.74 $p |- ( ( ps -> ph )
      -> ( ( ( ph \/ ps ) \/ ch ) -> ( ph \/ ch ) ) ) $=
    ( wi wo orel2 ax-1 ja orim1d ) BADABEZACBAJADBAFAJGHI $.

  $( Disjunction distributes over implication.  (Contributed by Wolf Lammen,
     5-Jan-2013.) $)
  orimdi $p |- ( ( ph \/ ( ps -> ch ) )
        <-> ( ( ph \/ ps ) -> ( ph \/ ch ) ) ) $=
    ( wn wi wo imdi df-or imbi12i 3bitr4i ) ADZBCEZEKBEZKCEZEALFABFZACFZEKBCGAL
    HOMPNABHACHIJ $.

  $( Theorem *2.76 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.76 $p |- ( ( ph \/ ( ps -> ch ) )
      -> ( ( ph \/ ps ) -> ( ph \/ ch ) ) ) $=
    ( wi wo orimdi biimpi ) ABCDEABEACEDABCFG $.

  $( Theorem *2.75 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 4-Jan-2013.) $)
  pm2.75 $p |- ( ( ph \/ ps )
       -> ( ( ph \/ ( ps -> ch ) ) -> ( ph \/ ch ) ) ) $=
    ( wi wo pm2.76 com12 ) ABCDEABEACEABCFG $.

  $( Theorem *2.8 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 5-Jan-2013.) $)
  pm2.8 $p |- ( ( ph \/ ps ) -> ( ( -. ps \/ ch ) -> ( ph \/ ch ) ) ) $=
    ( wo wn pm2.53 con1d orim1d ) ABDZBEACIABABFGH $.

  $( Theorem *2.81 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.81 $p |- ( ( ps -> ( ch -> th ) )
      -> ( ( ph \/ ps ) -> ( ( ph \/ ch ) -> ( ph \/ th ) ) ) ) $=
    ( wi wo orim2 pm2.76 syl6 ) BCDEZEABFAJFACFADFEABJGACDHI $.

  $( Theorem *2.82 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.) $)
  pm2.82 $p |- ( ( ( ph \/ ps ) \/ ch ) -> ( ( ( ph \/ -. ch ) \/ th )
      -> ( ( ph \/ ps ) \/ th ) ) ) $=
    ( wo wn wi ax-1 pm2.24 orim2d jaoi orim1d ) ABEZCEACFZEZMDMOMGCMOHCNBACBIJK
    L $.

  $( Theorem *2.85 of [WhiteheadRussell] p. 108.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 5-Jan-2013.) $)
  pm2.85 $p |- ( ( ( ph \/ ps ) -> ( ph \/ ch ) )
      -> ( ph \/ ( ps -> ch ) ) ) $=
    ( wi wo orimdi biimpri ) ABCDEABEACEDABCFG $.

  ${
    pm3.2ni.1 $e |- -. ph $.
    pm3.2ni.2 $e |- -. ps $.
    $( Infer negated disjunction of negated premises.  (Contributed by NM,
       4-Apr-1995.) $)
    pm3.2ni $p |- -. ( ph \/ ps ) $=
      ( wo id pm2.21i jaoi mto ) ABEACAABAFBADGHI $.
  $}

  $( Absorption of redundant internal disjunct.  Compare Theorem *4.45 of
     [WhiteheadRussell] p. 119.  (Contributed by NM, 21-Jun-1993.)  (Proof
     shortened by Wolf Lammen, 28-Feb-2014.) $)
  orabs $p |- ( ph <-> ( ( ph \/ ps ) /\ ph ) ) $=
    ( wo orc pm4.71ri ) AABCABDE $.

  $( Absorb a disjunct into a conjunct.  (Contributed by Roy F. Longton,
     23-Jun-2005.)  (Proof shortened by Wolf Lammen, 10-Nov-2013.) $)
  oranabs $p |- ( ( ( ph \/ -. ps ) /\ ps ) <-> ( ph /\ ps ) ) $=
    ( wn wo biortn orcom syl6rbb pm5.32ri ) BABCZDZABAIADJBAEIAFGH $.

  $( Two propositions are equivalent if they are both true.  Theorem *5.1 of
     [WhiteheadRussell] p. 123.  (Contributed by NM, 21-May-1994.) $)
  pm5.1 $p |- ( ( ph /\ ps ) -> ( ph <-> ps ) ) $=
    ( wb pm5.501 biimpa ) ABABCABDE $.

  $( Two propositions are equivalent if they are both false.  Theorem *5.21 of
     [WhiteheadRussell] p. 124.  (Contributed by NM, 21-May-1994.) $)
  pm5.21 $p |- ( ( -. ph /\ -. ps ) -> ( ph <-> ps ) ) $=
    ( wn wb pm5.21im imp ) ACBCABDABEF $.

  $( If neither of two propositions is true, then these propositions are
     equivalent.  (Contributed by BJ, 26-Apr-2019.) $)
  norbi $p |- ( -. ( ph \/ ps ) -> ( ph <-> ps ) ) $=
    ( wo orc olc pm5.21ni ) AABCBABDBAEF $.

  $( If two propositions are not equivalent, then at least one is true.
     (Contributed by BJ, 19-Apr-2019.)  (Proof shortened by Wolf Lammen,
     19-Jan-2020.) $)
  nbior $p |- ( -. ( ph <-> ps ) -> ( ph \/ ps ) ) $=
    ( wo wb norbi con1i ) ABCABDABEF $.

  $( Obsolete proof of ~ nbior as of 19-Jan-2020.  (Contributed by BJ,
     19-Apr-2019.)  (Proof modification is discouraged.)
     (New usage is discouraged.) $)
  nbiorOLD $p |- ( -. ( ph <-> ps ) -> ( ph \/ ps ) ) $=
    ( wb wn wa wo pm5.21 con3i oran sylibr ) ABCZDADBDEZDABFLKABGHABIJ $.

  $( Theorem *3.43 (Comp) of [WhiteheadRussell] p. 113.  (Contributed by NM,
     3-Jan-2005.) $)
  pm3.43 $p |- ( ( ( ph -> ps ) /\ ( ph -> ch ) )
      -> ( ph -> ( ps /\ ch ) ) ) $=
    ( wi wa pm3.43i imp ) ABDACDABCEDABCFG $.

  $( Distributive law for implication over conjunction.  Compare Theorem *4.76
     of [WhiteheadRussell] p. 121.  (Contributed by NM, 3-Apr-1994.)  (Proof
     shortened by Wolf Lammen, 27-Nov-2013.) $)
  jcab $p |- ( ( ph -> ( ps /\ ch ) )
      <-> ( ( ph -> ps ) /\ ( ph -> ch ) ) ) $=
    ( wa wi simpl imim2i simpr jca pm3.43 impbii ) ABCDZEZABEZACEZDMNOLBABCFGLC
    ABCHGIABCJK $.

  $( Distributive law for disjunction.  Theorem *4.41 of [WhiteheadRussell]
     p. 119.  (Contributed by NM, 5-Jan-1993.)  (Proof shortened by Andrew
     Salmon, 7-May-2011.)  (Proof shortened by Wolf Lammen, 28-Nov-2013.) $)
  ordi $p |- ( ( ph \/ ( ps /\ ch ) ) <-> ( ( ph \/ ps ) /\ ( ph \/ ch ) ) ) $=
    ( wn wa wi wo jcab df-or anbi12i 3bitr4i ) ADZBCEZFLBFZLCFZEAMGABGZACGZELBC
    HAMIPNQOABIACIJK $.

  $( Distributive law for disjunction.  (Contributed by NM, 12-Aug-1994.) $)
  ordir $p |- ( ( ( ph /\ ps ) \/ ch ) <->
              ( ( ph \/ ch ) /\ ( ps \/ ch ) ) ) $=
    ( wa wo ordi orcom anbi12i 3bitr4i ) CABDZECAEZCBEZDJCEACEZBCEZDCABFJCGMKNL
    ACGBCGHI $.

  $( Theorem *4.76 of [WhiteheadRussell] p. 121.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.76 $p |- ( ( ( ph -> ps ) /\ ( ph -> ch ) ) <->
                ( ph -> ( ps /\ ch ) ) ) $=
    ( wa wi jcab bicomi ) ABCDEABEACEDABCFG $.

  $( Distributive law for conjunction.  Theorem *4.4 of [WhiteheadRussell]
     p. 118.  (Contributed by NM, 21-Jun-1993.)  (Proof shortened by Wolf
     Lammen, 5-Jan-2013.) $)
  andi $p |- ( ( ph /\ ( ps \/ ch ) ) <-> ( ( ph /\ ps ) \/ ( ph /\ ch ) ) ) $=
    ( wo wa orc olc jaodan anim2i jaoi impbii ) ABCDZEZABEZACEZDZABPCNOFONGHNMO
    BLABCFICLACBGIJK $.

  $( Distributive law for conjunction.  (Contributed by NM, 12-Aug-1994.) $)
  andir $p |- ( ( ( ph \/ ps ) /\ ch ) <->
              ( ( ph /\ ch ) \/ ( ps /\ ch ) ) ) $=
    ( wo wa andi ancom orbi12i 3bitr4i ) CABDZECAEZCBEZDJCEACEZBCEZDCABFJCGMKNL
    ACGBCGHI $.

  $( Double distributive law for disjunction.  (Contributed by NM,
     12-Aug-1994.) $)
  orddi $p |- ( ( ( ph /\ ps ) \/ ( ch /\ th ) ) <->
              ( ( ( ph \/ ch ) /\ ( ph \/ th ) ) /\
              ( ( ps \/ ch ) /\ ( ps \/ th ) ) ) ) $=
    ( wa wo ordir ordi anbi12i bitri ) ABECDEZFAKFZBKFZEACFADFEZBCFBDFEZEABKGLN
    MOACDHBCDHIJ $.

  $( Double distributive law for conjunction.  (Contributed by NM,
     12-Aug-1994.) $)
  anddi $p |- ( ( ( ph \/ ps ) /\ ( ch \/ th ) ) <->
              ( ( ( ph /\ ch ) \/ ( ph /\ th ) ) \/
              ( ( ps /\ ch ) \/ ( ps /\ th ) ) ) ) $=
    ( wo wa andir andi orbi12i bitri ) ABECDEZFAKFZBKFZEACFADFEZBCFBDFEZEABKGLN
    MOACDHBCDHIJ $.

  $( Prove formula-building rules for the biconditional connective. $)

  $( Theorem *4.39 of [WhiteheadRussell] p. 118.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.39 $p |- ( ( ( ph <-> ch ) /\ ( ps <-> th ) ) ->
                ( ( ph \/ ps ) <-> ( ch \/ th ) ) ) $=
    ( wb wa simpl simpr orbi12d ) ACEZBDEZFACBDJKGJKHI $.

  $( Theorem *4.38 of [WhiteheadRussell] p. 118.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.38 $p |- ( ( ( ph <-> ch ) /\ ( ps <-> th ) ) ->
                ( ( ph /\ ps ) <-> ( ch /\ th ) ) ) $=
    ( wb wa simpl simpr anbi12d ) ACEZBDEZFACBDJKGJKHI $.

  ${
    bi2an9.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    bi2an9.2 $e |- ( th -> ( ta <-> et ) ) $.
    $( Deduction joining two equivalences to form equivalence of conjunctions.
       (Contributed by NM, 31-Jul-1995.) $)
    bi2anan9 $p |- ( ( ph /\ th ) -> ( ( ps /\ ta ) <-> ( ch /\ et ) ) ) $=
      ( wa anbi1d anbi2d sylan9bb ) ABEICEIDCFIABCEGJDEFCHKL $.

    $( Deduction joining two equivalences to form equivalence of conjunctions.
       (Contributed by NM, 19-Feb-1996.) $)
    bi2anan9r $p |- ( ( th /\ ph ) -> ( ( ps /\ ta ) <-> ( ch /\ et ) ) ) $=
      ( wa wb bi2anan9 ancoms ) ADBEICFIJABCDEFGHKL $.

    $( Deduction joining two biconditionals with different antecedents.
       (Contributed by NM, 12-May-2004.) $)
    bi2bian9 $p |- ( ( ph /\ th ) -> ( ( ps <-> ta ) <-> ( ch <-> et ) ) ) $=
      ( wa wb adantr adantl bibi12d ) ADIBCEFABCJDGKDEFJAHLM $.
  $}

  $( Implication in terms of biconditional and disjunction.  Theorem *4.72 of
     [WhiteheadRussell] p. 121.  (Contributed by NM, 30-Aug-1993.)  (Proof
     shortened by Wolf Lammen, 30-Jan-2013.) $)
  pm4.72 $p |- ( ( ph -> ps ) <-> ( ps <-> ( ph \/ ps ) ) ) $=
    ( wi wo wb olc pm2.621 impbid2 orc bi2 syl5 impbii ) ABCZBABDZEZMBNBAFABGHA
    NOBABIBNJKL $.

  $( Simplify an implication between implications.  (Contributed by Paul
     Chapman, 17-Nov-2012.)  (Proof shortened by Wolf Lammen, 3-Apr-2013.) $)
  imimorb $p |- ( ( ( ps -> ch ) -> ( ph -> ch ) ) <->
                  ( ph -> ( ps \/ ch ) ) ) $=
    ( wi wo bi2.04 dfor2 imbi2i bitr4i ) BCDZACDDAJCDZDABCEZDJACFLKABCGHI $.

  $( Theorem *5.33 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.33 $p |- ( ( ph /\ ( ps -> ch ) ) <->
                ( ph /\ ( ( ph /\ ps ) -> ch ) ) ) $=
    ( wi wa ibar imbi1d pm5.32i ) ABCDABEZCDABICABFGH $.

  $( Theorem *5.36 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.36 $p |- ( ( ph /\ ( ph <-> ps ) ) <-> ( ps /\ ( ph <-> ps ) ) ) $=
    ( wb id pm5.32ri ) ABCZABFDE $.

  ${
    bianabs.1 $e |- ( ph -> ( ps <-> ( ph /\ ch ) ) ) $.
    $( Absorb a hypothesis into the second member of a biconditional.
       (Contributed by FL, 15-Feb-2007.) $)
    bianabs $p |- ( ph -> ( ps <-> ch ) ) $=
      ( wa ibar bitr4d ) ABACECDACFG $.
  $}

  $( Absorption of disjunction into equivalence.  (Contributed by NM,
     6-Aug-1995.)  (Proof shortened by Wolf Lammen, 3-Nov-2013.) $)
  oibabs $p |- ( ( ( ph \/ ps ) -> ( ph <-> ps ) ) <-> ( ph <-> ps ) ) $=
    ( wo wb wi norbi id ja ax-1 impbii ) ABCZABDZELKLLABFLGHLKIJ $.

  $( Law of noncontradiction.  Theorem *3.24 of [WhiteheadRussell] p. 111 (who
     call it the "law of contradiction").  (Contributed by NM, 16-Sep-1993.)
     (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
  pm3.24 $p |- -. ( ph /\ -. ph ) $=
    ( wi wn wa id iman mpbi ) AABAACDCAEAAFG $.

  $( Theorem *2.26 of [WhiteheadRussell] p. 104.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 23-Nov-2012.) $)
  pm2.26 $p |- ( -. ph \/ ( ( ph -> ps ) -> ps ) ) $=
    ( wi pm2.27 imori ) AABCBCABDE $.

  $( Theorem *5.11 of [WhiteheadRussell] p. 123.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.11 $p |- ( ( ph -> ps ) \/ ( -. ph -> ps ) ) $=
    ( wi wn pm2.5 orri ) ABCADBCABEF $.

  $( Theorem *5.12 of [WhiteheadRussell] p. 123.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.12 $p |- ( ( ph -> ps ) \/ ( ph -> -. ps ) ) $=
    ( wi wn pm2.51 orri ) ABCABDCABEF $.

  $( Theorem *5.14 of [WhiteheadRussell] p. 123.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.14 $p |- ( ( ph -> ps ) \/ ( ps -> ch ) ) $=
    ( wi wn ax-1 con3i pm2.21d orri ) ABDZBCDJEBCBJBAFGHI $.

  $( Theorem *5.13 of [WhiteheadRussell] p. 123.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 14-Nov-2012.) $)
  pm5.13 $p |- ( ( ph -> ps ) \/ ( ps -> ph ) ) $=
    ( pm5.14 ) ABAC $.

  $( Theorem *5.17 of [WhiteheadRussell] p. 124.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 3-Jan-2013.) $)
  pm5.17 $p |- ( ( ( ph \/ ps ) /\ -. ( ph /\ ps ) ) <-> ( ph <-> -. ps ) ) $=
    ( wn wb wi wa wo bicom dfbi2 orcom df-or bitr2i imnan anbi12i 3bitrri ) ABC
    ZDPADPAEZAPEZFABGZABFCZFAPHPAIQSRTSBAGQABJBAKLABMNO $.

  $( Theorem *5.15 of [WhiteheadRussell] p. 124.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 15-Oct-2013.) $)
  pm5.15 $p |- ( ( ph <-> ps ) \/ ( ph <-> -. ps ) ) $=
    ( wb wn xor3 biimpi orri ) ABCZABDCZHDIABEFG $.

  $( Theorem *5.16 of [WhiteheadRussell] p. 124.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 17-Oct-2013.) $)
  pm5.16 $p |- -. ( ( ph <-> ps ) /\ ( ph <-> -. ps ) ) $=
    ( wb wn wi wa pm5.18 biimpi imnan mpbi ) ABCZABDCZDZEKLFDKMABGHKLIJ $.

  $( Two ways to express "exclusive or."  Theorem *5.22 of [WhiteheadRussell]
     p. 124.  (Contributed by NM, 3-Jan-2005.)  (Proof shortened by Wolf
     Lammen, 22-Jan-2013.) $)
  xor $p |- ( -. ( ph <-> ps ) <->
                ( ( ph /\ -. ps ) \/ ( ps /\ -. ph ) ) ) $=
    ( wn wa wo wb wi iman anbi12i dfbi2 ioran 3bitr4ri con1bii ) ABCDZBACDZEZAB
    FZABGZBAGZDNCZOCZDQPCRTSUAABHBAHIABJNOKLM $.

  $( Two ways to express "exclusive or."  (Contributed by NM, 3-Jan-2005.)
     (Proof shortened by Wolf Lammen, 24-Jan-2013.) $)
  nbi2 $p |- ( -. ( ph <-> ps ) <-> ( ( ph \/ ps ) /\ -. ( ph /\ ps ) ) ) $=
    ( wb wn wo wa xor3 pm5.17 bitr4i ) ABCDABDCABEABFDFABGABHI $.

  $( An alternate definition of the biconditional.  Theorem *5.23 of
     [WhiteheadRussell] p. 124.  (Contributed by NM, 27-Jun-2002.)  (Proof
     shortened by Wolf Lammen, 3-Nov-2013.) $)
  dfbi3 $p |- ( ( ph <-> ps ) <-> ( ( ph /\ ps ) \/ ( -. ph /\ -. ps ) ) ) $=
    ( wn wb wa wo xor pm5.18 notnot anbi2i ancom orbi12i 3bitr4i ) ABCZDCANCZEZ
    NACZEZFABDABEZQNEZFANGABHSPTRBOABIJQNKLM $.

  $( Theorem *5.24 of [WhiteheadRussell] p. 124.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.24 $p |- ( -. ( ( ph /\ ps ) \/ ( -. ph /\ -. ps ) ) <->
                ( ( ph /\ -. ps ) \/ ( ps /\ -. ph ) ) ) $=
    ( wb wn wa wo xor dfbi3 xchnxbi ) ABCABDZEBADZEFABEKJEFABGABHI $.

  $( Conjunction distributes over exclusive-or, using ` -. ( ph <-> ps ) ` to
     express exclusive-or.  This is one way to interpret the distributive law
     of multiplication over addition in modulo 2 arithmetic.  This is not
     necessarily true in intuitionistic logic, though ~ anxordi does hold in
     it.  (Contributed by NM, 3-Oct-2008.) $)
  xordi $p |- ( ( ph /\ -. ( ps <-> ch ) ) <->
                -. ( ( ph /\ ps ) <-> ( ph /\ ch ) ) ) $=
    ( wb wn wa wi annim pm5.32 xchbinx ) ABCDZEFAKGABFACFDAKHABCIJ $.

  $( A wff disjoined with truth is true.  (Contributed by NM, 23-May-1999.) $)
  biort $p |- ( ph -> ( ph <-> ( ph \/ ps ) ) ) $=
    ( wo orc ax-1 impbid2 ) AAABCZABDAGEF $.

  $( Theorem *5.55 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 20-Jan-2013.) $)
  pm5.55 $p |- ( ( ( ph \/ ps ) <-> ph ) \/ ( ( ph \/ ps ) <-> ps ) ) $=
    ( wo wb biort bicomd wn biorf nsyl4 con1i orri ) ABCZADZLBDZNMAMNAALABEFAGB
    LABHFIJK $.

  $( Selecting one statement from a disjunction if one of the disjuncted
     statements is false.  (Contributed by AV, 6-Sep-2018.)  (Proof shortened
     by AV, 13-Oct-2018.)  (Proof shortened by Wolf Lammen, 19-Jan-2020.) $)
  ornld $p |- ( ph -> ( ( ( ph -> ( th \/ ta ) ) /\ -. th ) -> ta ) ) $=
    ( wo wi wn wa pm3.35 ord expimpd ) AABCDZEZBFCALGBCAKHIJ $.

  $( Obsolete proof of ~ ornld as of 19-Jan-2020.  (Contributed by AV,
     6-Sep-2018.)  (Proof shortened by AV, 13-Oct-2018.)
     (Proof modification is discouraged.)  (New usage is discouraged.) $)
  ornldOLD $p |- ( ph -> ( ( ( ph -> ( th \/ ta ) ) /\ -. th ) -> ta ) ) $=
    ( wo wi wn wa orel1 imim2d impcom com12 ) ABCDZEZBFZGACNMACENLCABCHIJK $.

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Miscellaneous theorems of propositional calculus
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
$)

  ${
    pm5.21nd.1 $e |- ( ( ph /\ ps ) -> th ) $.
    pm5.21nd.2 $e |- ( ( ph /\ ch ) -> th ) $.
    pm5.21nd.3 $e |- ( th -> ( ps <-> ch ) ) $.
    $( Eliminate an antecedent implied by each side of a biconditional.
       (Contributed by NM, 20-Nov-2005.)  (Proof shortened by Wolf Lammen,
       4-Nov-2013.) $)
    pm5.21nd $p |- ( ph -> ( ps <-> ch ) ) $=
      ( ex wb wi a1i pm5.21ndd ) ADBCABDEHACDFHDBCIJAGKL $.
  $}

  $( Theorem *5.35 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.35 $p |- ( ( ( ph -> ps ) /\ ( ph -> ch ) ) ->
                ( ph -> ( ps <-> ch ) ) ) $=
    ( wi wa pm5.1 pm5.74rd ) ABDZACDZEABCHIFG $.

  $( Theorem *5.54 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 7-Nov-2013.) $)
  pm5.54 $p |- ( ( ( ph /\ ps ) <-> ph ) \/ ( ( ph /\ ps ) <-> ps ) ) $=
    ( wa wb iba bicomd adantl pm5.21ni orri ) ABCZADZJBDJKBBKABAJBAEFZGLHI $.

  ${
    baib.1 $e |- ( ph <-> ( ps /\ ch ) ) $.
    $( Move conjunction outside of biconditional.  (Contributed by NM,
       13-May-1999.) $)
    baib $p |- ( ps -> ( ph <-> ch ) ) $=
      ( wa ibar syl6rbbr ) BCBCEABCFDG $.

    $( Move conjunction outside of biconditional.  (Contributed by NM,
       11-Jul-1994.) $)
    baibr $p |- ( ps -> ( ch <-> ph ) ) $=
      ( baib bicomd ) BACABCDEF $.

    $( Move conjunction outside of biconditional.  (Contributed by Mario
       Carneiro, 11-Sep-2015.)  (Proof shortened by Wolf Lammen,
       19-Jan-2020.) $)
    rbaibr $p |- ( ch -> ( ps <-> ph ) ) $=
      ( wa iba syl6bbr ) CBBCEACBFDG $.

    $( Move conjunction outside of biconditional.  (Contributed by Mario
       Carneiro, 11-Sep-2015.)  (Proof shortened by Wolf Lammen,
       19-Jan-2020.) $)
    rbaib $p |- ( ch -> ( ph <-> ps ) ) $=
      ( rbaibr bicomd ) CBAABCDEF $.

    $( Obsolete proof of ~ rbaib as of 19-Jan-2020.  (Contributed by Mario
       Carneiro, 11-Sep-2015.)  (Proof modification is discouraged.)
       (New usage is discouraged.) $)
    rbaibOLD $p |- ( ch -> ( ph <-> ps ) ) $=
      ( wa ancom bitri baib ) ACBABCECBEDBCFGH $.

    $( Obsolete proof of ~ rbaibr as of 19-Jan-2020.  (Contributed by Mario
       Carneiro, 11-Sep-2015.)  (Proof modification is discouraged.)
       (New usage is discouraged.) $)
    rbaibrOLD $p |- ( ch -> ( ps <-> ph ) ) $=
      ( wa ancom bitri baibr ) ACBABCECBEDBCFGH $.
  $}

  ${
    baibd.1 $e |- ( ph -> ( ps <-> ( ch /\ th ) ) ) $.
    $( Move conjunction outside of biconditional.  (Contributed by Mario
       Carneiro, 11-Sep-2015.) $)
    baibd $p |- ( ( ph /\ ch ) -> ( ps <-> th ) ) $=
      ( wa ibar bicomd sylan9bb ) ABCDFZCDECDJCDGHI $.

    $( Move conjunction outside of biconditional.  (Contributed by Mario
       Carneiro, 11-Sep-2015.) $)
    rbaibd $p |- ( ( ph /\ th ) -> ( ps <-> ch ) ) $=
      ( wa iba bicomd sylan9bb ) ABCDFZDCEDCJDCGHI $.
  $}

  $( Theorem *5.44 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.) $)
  pm5.44 $p |- ( ( ph -> ps ) -> ( ( ph -> ch ) <->
                ( ph -> ( ps /\ ch ) ) ) ) $=
    ( wa wi jcab baibr ) ABCDEABEACEABCFG $.

  $( Conjunction in antecedent versus disjunction in consequent.  Theorem *5.6
     of [WhiteheadRussell] p. 125.  (Contributed by NM, 8-Jun-1994.) $)
  pm5.6 $p |- ( ( ( ph /\ -. ps ) -> ch ) <-> ( ph -> ( ps \/ ch ) ) ) $=
    ( wn wa wi wo impexp df-or imbi2i bitr4i ) ABDZECFALCFZFABCGZFALCHNMABCIJK
    $.

  ${
    orcanai.1 $e |- ( ph -> ( ps \/ ch ) ) $.
    $( Change disjunction in consequent to conjunction in antecedent.
       (Contributed by NM, 8-Jun-1994.) $)
    orcanai $p |- ( ( ph /\ -. ps ) -> ch ) $=
      ( wn ord imp ) ABECABCDFG $.
  $}

  ${
    intnan.1 $e |- -. ph $.
    $( Introduction of conjunct inside of a contradiction.  (Contributed by NM,
       16-Sep-1993.) $)
    intnan $p |- -. ( ps /\ ph ) $=
      ( wa simpr mto ) BADACBAEF $.

    $( Introduction of conjunct inside of a contradiction.  (Contributed by NM,
       3-Apr-1995.) $)
    intnanr $p |- -. ( ph /\ ps ) $=
      ( wa simpl mto ) ABDACABEF $.
  $}

  ${
    intnand.1 $e |- ( ph -> -. ps ) $.
    $( Introduction of conjunct inside of a contradiction.  (Contributed by NM,
       10-Jul-2005.) $)
    intnand $p |- ( ph -> -. ( ch /\ ps ) ) $=
      ( wa simpr nsyl ) ABCBEDCBFG $.

    $( Introduction of conjunct inside of a contradiction.  (Contributed by NM,
       10-Jul-2005.) $)
    intnanrd $p |- ( ph -> -. ( ps /\ ch ) ) $=
      ( wa simpl nsyl ) ABBCEDBCFG $.
  $}

  ${
    mpbiran.1 $e |- ps $.
    mpbiran.2 $e |- ( ph <-> ( ps /\ ch ) ) $.
    $( Detach truth from conjunction in biconditional.  (Contributed by NM,
       27-Feb-1996.) $)
    mpbiran $p |- ( ph <-> ch ) $=
      ( wa biantrur bitr4i ) ABCFCEBCDGH $.
  $}

  ${
    mpbiran2.1 $e |- ch $.
    mpbiran2.2 $e |- ( ph <-> ( ps /\ ch ) ) $.
    $( Detach truth from conjunction in biconditional.  (Contributed by NM,
       22-Feb-1996.) $)
    mpbiran2 $p |- ( ph <-> ps ) $=
      ( wa biantru bitr4i ) ABCFBECBDGH $.
  $}

  ${
    mpbir2an.1 $e |- ps $.
    mpbir2an.2 $e |- ch $.
    mpbiran2an.1 $e |- ( ph <-> ( ps /\ ch ) ) $.
    $( Detach a conjunction of truths in a biconditional.  (Contributed by NM,
       10-May-2005.) $)
    mpbir2an $p |- ph $=
      ( mpbiran mpbir ) ACEABCDFGH $.
  $}

  ${
    mpbi2and.1 $e |- ( ph -> ps ) $.
    mpbi2and.2 $e |- ( ph -> ch ) $.
    mpbi2and.3 $e |- ( ph -> ( ( ps /\ ch ) <-> th ) ) $.
    $( Detach a conjunction of truths in a biconditional.  (Contributed by NM,
       6-Nov-2011.)  (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
    mpbi2and $p |- ( ph -> th ) $=
      ( wa jca mpbid ) ABCHDABCEFIGJ $.
  $}

  ${
    mpbir2and.1 $e |- ( ph -> ch ) $.
    mpbir2and.2 $e |- ( ph -> th ) $.
    mpbir2and.3 $e |- ( ph -> ( ps <-> ( ch /\ th ) ) ) $.
    $( Detach a conjunction of truths in a biconditional.  (Contributed by NM,
       6-Nov-2011.)  (Proof shortened by Wolf Lammen, 24-Nov-2012.) $)
    mpbir2and $p |- ( ph -> ps ) $=
      ( wa jca mpbird ) ABCDHACDEFIGJ $.
  $}

  $( Theorem *5.62 of [WhiteheadRussell] p. 125.  (Contributed by Roy F.
     Longton, 21-Jun-2005.) $)
  pm5.62 $p |- ( ( ( ph /\ ps ) \/ -. ps ) <-> ( ph \/ -. ps ) ) $=
    ( wa wn wo exmid ordir mpbiran2 ) ABCBDZEAIEBIEBFABIGH $.

  $( Theorem *5.63 of [WhiteheadRussell] p. 125.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 25-Dec-2012.) $)
  pm5.63 $p |- ( ( ph \/ ps ) <-> ( ph \/ ( -. ph /\ ps ) ) ) $=
    ( wn wa wo exmid ordi mpbiran bicomi ) AACZBDEZABEZKAJELAFAJBGHI $.

  ${
    bianfi.1 $e |- -. ph $.
    $( A wff conjoined with falsehood is false.  (Contributed by NM,
       21-Jun-1993.)  (Proof shortened by Wolf Lammen, 26-Nov-2012.) $)
    bianfi $p |- ( ph <-> ( ps /\ ph ) ) $=
      ( wa intnan 2false ) ABADCABCEF $.
  $}

  ${
    bianfd.1 $e |- ( ph -> -. ps ) $.
    $( A wff conjoined with falsehood is false.  (Contributed by NM,
       27-Mar-1995.)  (Proof shortened by Wolf Lammen, 5-Nov-2013.) $)
    bianfd $p |- ( ph -> ( ps <-> ( ps /\ ch ) ) ) $=
      ( wa intnanrd 2falsed ) ABBCEDABCDFG $.
  $}

  $( Theorem *4.43 of [WhiteheadRussell] p. 119.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Wolf Lammen, 26-Nov-2012.) $)
  pm4.43 $p |- ( ph <-> ( ( ph \/ ps ) /\ ( ph \/ -. ps ) ) ) $=
    ( wn wa wo pm3.24 biorfi ordi bitri ) AABBCZDZEABEAJEDKABFGABJHI $.

  $( Theorem *4.82 of [WhiteheadRussell] p. 122.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.82 $p |- ( ( ( ph -> ps ) /\ ( ph -> -. ps ) ) <-> -. ph ) $=
    ( wi wn wa pm2.65 imp pm2.21 jca impbii ) ABCZABDZCZEADZKMNABFGNKMABHALHIJ
    $.

  $( Theorem *4.83 of [WhiteheadRussell] p. 122.  (Contributed by NM,
     3-Jan-2005.) $)
  pm4.83 $p |- ( ( ( ph -> ps ) /\ ( -. ph -> ps ) ) <-> ps ) $=
    ( wn wo wi wa exmid a1bi jaob bitr2i ) BAACZDZBEABEKBEFLBAGHABKIJ $.

  $( Negation inferred from embedded conjunct.  (Contributed by NM,
     20-Aug-1993.)  (Proof shortened by Wolf Lammen, 25-Nov-2012.) $)
  pclem6 $p |- ( ( ph <-> ( ps /\ -. ph ) ) -> -. ps ) $=
    ( wn wa wb ibar nbbn sylib con2i ) BABACZDZEZBJKELCBJFAKGHI $.

  $( A transitive law of equivalence.  Compare Theorem *4.22 of
     [WhiteheadRussell] p. 117.  (Contributed by NM, 18-Aug-1993.) $)
  biantr $p |- ( ( ( ph <-> ps ) /\ ( ch <-> ps ) ) -> ( ph <-> ch ) ) $=
    ( wb id bibi2d biimparc ) CBDZACDABDHCBAHEFG $.

  $( Disjunction distributes over the biconditional.  An axiom of system DS in
     Vladimir Lifschitz, "On calculational proofs" (1998),
     ~ http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.25.3384 .
     (Contributed by NM, 8-Jan-2005.)  (Proof shortened by Wolf Lammen,
     4-Feb-2013.) $)
  orbidi $p |- ( ( ph \/ ( ps <-> ch ) ) <->
                ( ( ph \/ ps ) <-> ( ph \/ ch ) ) ) $=
    ( wn wb wi wo pm5.74 df-or bibi12i 3bitr4i ) ADZBCEZFLBFZLCFZEAMGABGZACGZEL
    BCHAMIPNQOABIACIJK $.

  $( Lukasiewicz's shortest axiom for equivalential calculus.  Storrs McCall,
     ed., _Polish Logic 1920-1939_ (Oxford, 1967), p. 96.  (Contributed by NM,
     10-Jan-2005.) $)
  biluk $p |- ( ( ph <-> ps ) <-> ( ( ch <-> ps ) <-> ( ph <-> ch ) ) ) $=
    ( wb bicom bibi1i biass bitri mpbi bitr4i ) ABDZCBACDZDZDZCBDLDKCDZMDKNDOBA
    DZCDMKPCABEFBACGHKCMGICBLGJ $.

  $( Disjunction distributes over the biconditional.  Theorem *5.7 of
     [WhiteheadRussell] p. 125.  This theorem is similar to ~ orbidi .
     (Contributed by Roy F. Longton, 21-Jun-2005.) $)
  pm5.7 $p |- ( ( ( ph \/ ch ) <-> ( ps \/ ch ) ) <->
               ( ch \/ ( ph <-> ps ) ) ) $=
    ( wb wo orbidi orcom bibi12i bitr2i ) CABDECAEZCBEZDACEZBCEZDCABFJLKMCAGCBG
    HI $.

  $( Dijkstra-Scholten's Golden Rule for calculational proofs.  (Contributed by
     NM, 10-Jan-2005.) $)
  bigolden $p |- ( ( ( ph /\ ps ) <-> ph ) <-> ( ps <-> ( ph \/ ps ) ) ) $=
    ( wi wa wb wo pm4.71 pm4.72 bicom 3bitr3ri ) ABCAABDZEBABFEKAEABGABHAKIJ $.

  $( Theorem *5.71 of [WhiteheadRussell] p. 125.  (Contributed by Roy F.
     Longton, 23-Jun-2005.) $)
  pm5.71 $p |- ( ( ps -> -. ch ) -> ( ( ( ph \/ ps ) /\ ch ) <->
                ( ph /\ ch ) ) ) $=
    ( wn wo wa wb orel2 orc impbid1 anbi1d pm2.21 pm5.32rd ja ) BCDZABEZCFACFGB
    DZPACQPABAHABIJKOCPACPAGLMN $.

  $( Theorem *5.75 of [WhiteheadRussell] p. 126.  (Contributed by NM,
     3-Jan-2005.)  (Proof shortened by Andrew Salmon, 7-May-2011.)  (Proof
     shortened by Wolf Lammen, 23-Dec-2012.) $)
  pm5.75 $p |- ( ( ( ch -> -. ps ) /\ ( ph <-> ( ps \/ ch ) ) ) ->
                ( ( ph /\ -. ps ) <-> ch ) ) $=
    ( wo wb wn wa wi anbi1 anbi1i pm5.61 syl6bb pm4.71 biimpi bicomd sylan9bbr
    orcom bitri ) ABCDZEZABFZGZCUAGZCUAHZCTUBSUAGZUCASUAIUECBDZUAGUCSUFUABCQJCB
    KRLUDCUCUDCUCECUAMNOP $.

  $( Removal of conjunct from one side of an equivalence.  (Contributed by NM,
     21-Jun-1993.) $)
  bimsc1 $p |- ( ( ( ph -> ps ) /\ ( ch <-> ( ps /\ ph ) ) )
               -> ( ch <-> ph ) ) $=
    ( wi wa wb simpr ancr impbid2 bibi2d biimpa ) ABDZCBAEZFCAFLMACLMABAGABHIJK
    $.

  $( The disjunction of the four possible combinations of two wffs and their
     negations is always true.  (Contributed by David Abernethy,
     28-Jan-2014.) $)
  4exmid $p |- ( ( ( ph /\ ps ) \/ ( -. ph /\ -. ps ) )
              \/ ( ( ph /\ -. ps ) \/ ( ps /\ -. ph ) ) ) $=
    ( wb wn wo wa exmid dfbi3 xor orbi12i mpbi ) ABCZLDZEABFADZBDZFEZAOFBNFEZEL
    GLPMQABHABIJK $.

  ${
    ecase2d.1 $e |- ( ph -> ps ) $.
    ecase2d.2 $e |- ( ph -> -. ( ps /\ ch ) ) $.
    ecase2d.3 $e |- ( ph -> -. ( ps /\ th ) ) $.
    ecase2d.4 $e |- ( ph -> ( ta \/ ( ch \/ th ) ) ) $.
    $( Deduction for elimination by cases.  (Contributed by NM, 21-Apr-1994.)
       (Proof shortened by Wolf Lammen, 22-Dec-2012.) $)
    ecase2d $p |- ( ph -> ta ) $=
      ( wo idd wa pm2.21d mpand jaod mpjaod ) AEECDJAEKACEDABCEFABCLEGMNABDEFAB
      DLEHMNOIP $.
  $}

  ${
    ecase3.1 $e |- ( ph -> ch ) $.
    ecase3.2 $e |- ( ps -> ch ) $.
    ecase3.3 $e |- ( -. ( ph \/ ps ) -> ch ) $.
    $( Inference for elimination by cases.  (Contributed by NM, 23-Mar-1995.)
       (Proof shortened by Wolf Lammen, 26-Nov-2012.) $)
    ecase3 $p |- ch $=
      ( wo jaoi pm2.61i ) ABGCACBDEHFI $.
  $}

  ${
    ecase.1 $e |- ( -. ph -> ch ) $.
    ecase.2 $e |- ( -. ps -> ch ) $.
    ecase.3 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( Inference for elimination by cases.  (Contributed by NM,
       13-Jul-2005.) $)
    ecase $p |- ch $=
      ( ex pm2.61nii ) ABCABCFGDEH $.
  $}

  ${
    ecase3d.1 $e |- ( ph -> ( ps -> th ) ) $.
    ecase3d.2 $e |- ( ph -> ( ch -> th ) ) $.
    ecase3d.3 $e |- ( ph -> ( -. ( ps \/ ch ) -> th ) ) $.
    $( Deduction for elimination by cases.  (Contributed by NM, 2-May-1996.)
       (Proof shortened by Andrew Salmon, 7-May-2011.) $)
    ecase3d $p |- ( ph -> th ) $=
      ( wo jaod pm2.61d ) ABCHDABDCEFIGJ $.
  $}

  ${
    ecased.1 $e |- ( ph -> ( -. ps -> th ) ) $.
    ecased.2 $e |- ( ph -> ( -. ch -> th ) ) $.
    ecased.3 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( Deduction for elimination by cases.  (Contributed by NM, 8-Oct-2012.) $)
    ecased $p |- ( ph -> th ) $=
      ( wn wo wa pm3.11 syl5 ecase3d ) ABHZCHZDEFNOIHBCJADBCKGLM $.
  $}

  ${
    ecase3ad.1 $e |- ( ph -> ( ps -> th ) ) $.
    ecase3ad.2 $e |- ( ph -> ( ch -> th ) ) $.
    ecase3ad.3 $e |- ( ph -> ( ( -. ps /\ -. ch ) -> th ) ) $.
    $( Deduction for elimination by cases.  (Contributed by NM,
       24-May-2013.) $)
    ecase3ad $p |- ( ph -> th ) $=
      ( wn notnot2 syl5 ecased ) ABHZCHZDLHBADBIEJMHCADCIFJGK $.
  $}

  ${
    ccase.1 $e |- ( ( ph /\ ps ) -> ta ) $.
    ccase.2 $e |- ( ( ch /\ ps ) -> ta ) $.
    ccase.3 $e |- ( ( ph /\ th ) -> ta ) $.
    ccase.4 $e |- ( ( ch /\ th ) -> ta ) $.
    $( Inference for combining cases.  (Contributed by NM, 29-Jul-1999.)
       (Proof shortened by Wolf Lammen, 6-Jan-2013.) $)
    ccase $p |- ( ( ( ph \/ ch ) /\ ( ps \/ th ) ) -> ta ) $=
      ( wo jaoian jaodan ) ACJBEDABECFGKADECHIKL $.
  $}

  ${
    ccased.1 $e |- ( ph -> ( ( ps /\ ch ) -> et ) ) $.
    ccased.2 $e |- ( ph -> ( ( th /\ ch ) -> et ) ) $.
    ccased.3 $e |- ( ph -> ( ( ps /\ ta ) -> et ) ) $.
    ccased.4 $e |- ( ph -> ( ( th /\ ta ) -> et ) ) $.
    $( Deduction for combining cases.  (Contributed by NM, 9-May-2004.) $)
    ccased $p |- ( ph -> ( ( ( ps \/ th ) /\ ( ch \/ ta ) ) -> et ) ) $=
      ( wo wa wi com12 ccase ) BDKCEKLAFBCDEAFMABCLFGNADCLFHNABELFINADELFJNON
      $.
  $}

  ${
    ccase2.1 $e |- ( ( ph /\ ps ) -> ta ) $.
    ccase2.2 $e |- ( ch -> ta ) $.
    ccase2.3 $e |- ( th -> ta ) $.
    $( Inference for combining cases.  (Contributed by NM, 29-Jul-1999.) $)
    ccase2 $p |- ( ( ( ph \/ ch ) /\ ( ps \/ th ) ) -> ta ) $=
      ( adantr adantl ccase ) ABCDEFCEBGIDEAHJDECHJK $.
  $}

  ${
    4cases.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    4cases.2 $e |- ( ( ph /\ -. ps ) -> ch ) $.
    4cases.3 $e |- ( ( -. ph /\ ps ) -> ch ) $.
    4cases.4 $e |- ( ( -. ph /\ -. ps ) -> ch ) $.
    $( Inference eliminating two antecedents from the four possible cases that
       result from their true/false combinations.  (Contributed by NM,
       25-Oct-2003.) $)
    4cases $p |- ch $=
      ( pm2.61ian wn pm2.61i ) BCABCDFHABICEGHJ $.
  $}

  ${
    4casesdan.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    4casesdan.2 $e |- ( ( ph /\ ( ps /\ -. ch ) ) -> th ) $.
    4casesdan.3 $e |- ( ( ph /\ ( -. ps /\ ch ) ) -> th ) $.
    4casesdan.4 $e |- ( ( ph /\ ( -. ps /\ -. ch ) ) -> th ) $.
    $( Deduction eliminating two antecedents from the four possible cases that
       result from their true/false combinations.  (Contributed by NM,
       19-Mar-2013.) $)
    4casesdan $p |- ( ph -> th ) $=
      ( wi wa expcom wn 4cases ) BCADIABCJDEKABCLZJDFKABLZCJDGKAONJDHKM $.
  $}

  ${
    niabn.1 $e |- ph $.
    $( Miscellaneous inference relating falsehoods.  (Contributed by NM,
       31-Mar-1994.) $)
    niabn $p |- ( -. ps -> ( ( ch /\ ps ) <-> -. ph ) ) $=
      ( wa wn simpr pm2.24i pm5.21ni ) CBEBAFCBGABDHI $.
  $}

  $( Lemma for an alternate version of weak deduction theorem.  (Contributed by
     NM, 2-Apr-1994.)  (Proof shortened by Andrew Salmon, 7-May-2011.)  (Proof
     shortened by Wolf Lammen, 4-Dec-2012.) $)
  dedlem0a $p |- ( ph -> ( ps <-> ( ( ch -> ph ) -> ( ps /\ ph ) ) ) ) $=
    ( wa wi iba wb ax-1 biimt syl bitrd ) ABBADZCAEZLEZABFAMLNGACHMLIJK $.

  $( Lemma for an alternate version of weak deduction theorem.  (Contributed by
     NM, 2-Apr-1994.) $)
  dedlem0b $p |- ( -. ph -> ( ps <-> ( ( ps -> ph ) -> ( ch /\ ph ) ) ) ) $=
    ( wn wi wa pm2.21 imim2d com23 simpr imim12i con1d com12 impbid ) ADZBBAEZC
    AFZEZOPBQOAQBAQGHIROBRBABDPQABAGCAJKLMN $.

  $( Lemma for weak deduction theorem.  (Contributed by NM, 26-Jun-2002.)
     (Proof shortened by Andrew Salmon, 7-May-2011.) $)
  dedlema $p |- ( ph -> ( ps <-> ( ( ps /\ ph ) \/ ( ch /\ -. ph ) ) ) ) $=
    ( wa wn wo orc expcom wi simpl a1i pm2.24 adantld jaod impbid ) ABBADZCAEZD
    ZFZBASPRGHAPBRPBIABAJKAQBCABLMNO $.

  $( Lemma for weak deduction theorem.  (Contributed by NM, 15-May-1999.)
     (Proof shortened by Andrew Salmon, 7-May-2011.) $)
  dedlemb $p |- ( -. ph -> ( ch <-> ( ( ps /\ ph ) \/ ( ch /\ -. ph ) ) ) ) $=
    ( wn wa wo olc expcom pm2.21 adantld wi simpl a1i jaod impbid ) ADZCBAEZCPE
    ZFZCPSRQGHPQCRPACBACIJRCKPCPLMNO $.

  ${
    elimh.1 $e |- ( ( ph <-> ( ( ph /\ ch ) \/ ( ps /\ -. ch ) ) )
                     -> ( ch <-> ta ) ) $.
    elimh.2 $e |- ( ( ps <-> ( ( ph /\ ch ) \/ ( ps /\ -. ch ) ) )
                     -> ( th <-> ta ) ) $.
    elimh.3 $e |- th $.
    $( Hypothesis builder for weak deduction theorem.  For more information,
       see the Deduction Theorem link on the Metamath Proof Explorer home
       page.  (Contributed by NM, 26-Jun-2002.) $)
    elimh $p |- ta $=
      ( wa wn wo wb dedlema syl ibi dedlemb mpbii pm2.61i ) CECECAACIBCJZIKZLCE
      LCABMFNOSDEHSBTLDELCABPGNQR $.
  $}

  ${
    dedt.1 $e |- ( ( ph <-> ( ( ph /\ ch ) \/ ( ps /\ -. ch ) ) )
                     -> ( th <-> ta ) ) $.
    dedt.2 $e |- ta $.
    $( The weak deduction theorem.  For more information, see the Deduction
       Theorem link on the Metamath Proof Explorer home page.  (Contributed by
       NM, 26-Jun-2002.) $)
    dedt $p |- ( ch -> th ) $=
      ( wa wn wo wb dedlema mpbiri syl ) CAACHBCIHJKZDCABLODEGFMN $.
  $}

  $( Contraposition.  Theorem *2.16 of [WhiteheadRussell] p. 103.  This version
     of ~ con3 demonstrates the use of the weak deduction theorem ~ dedt to
     derive it from ~ con3i .  (Contributed by NM, 27-Jun-2002.)
     (Proof modification is discouraged.)  (New usage is discouraged.) $)
  con3th $p |- ( ( ph -> ps ) -> ( -. ps -> -. ph ) ) $=
    ( wi wn wa wo wb id notbid imbi1d imbi2d elimh con3i dedt ) BAABCZBDZADZCBO
    EAODEFZDZQCBRGZPSQTBRTHZIJARBAOAACARCTBRAUAKARGZARAUBHKAHLMN $.

  $( The consensus theorem.  This theorem and its dual (with ` \/ ` and ` /\ `
     interchanged) are commonly used in computer logic design to eliminate
     redundant terms from Boolean expressions.  Specifically, we prove that the
     term ` ( ps /\ ch ) ` on the left-hand side is redundant.  (Contributed by
     NM, 16-May-2003.)  (Proof shortened by Andrew Salmon, 13-May-2011.)
     (Proof shortened by Wolf Lammen, 20-Jan-2013.) $)
  consensus $p |- ( ( ( ( ph /\ ps ) \/ ( -. ph /\ ch ) ) \/ ( ps /\ ch ) ) <->
                      ( ( ph /\ ps ) \/ ( -. ph /\ ch ) ) ) $=
    ( wa wn wo id orc adantrr olc adantrl pm2.61ian jaoi impbii ) ABDZAEZCDZFZB
    CDZFRRRSRGASRABRCOQHIPCRBQOJKLMRSHN $.

  $( Theorem *4.42 of [WhiteheadRussell] p. 119.  (Contributed by Roy F.
     Longton, 21-Jun-2005.) $)
  pm4.42 $p |- ( ph <-> ( ( ph /\ ps ) \/ ( ph /\ -. ps ) ) ) $=
    ( wa wn wo wb dedlema dedlemb pm2.61i ) BAABCABDCEFBAAGBAAHI $.

  ${
    ninba.1 $e |- ph $.
    $( Miscellaneous inference relating falsehoods.  (Contributed by NM,
       31-Mar-1994.) $)
    ninba $p |- ( -. ps -> ( -. ph <-> ( ch /\ ps ) ) ) $=
      ( wn wa niabn bicomd ) BECBFAEABCDGH $.
  $}

  ${
    prlem1.1 $e |- ( ph -> ( et <-> ch ) ) $.
    prlem1.2 $e |- ( ps -> -. th ) $.
    $( A specialized lemma for set theory (to derive the Axiom of Pairing).
       (Contributed by NM, 18-Oct-1995.)  (Proof shortened by Andrew Salmon,
       13-May-2011.)  (Proof shortened by Wolf Lammen, 5-Jan-2013.) $)
    prlem1 $p |- ( ph -> ( ps ->
                  ( ( ( ps /\ ch ) \/ ( th /\ ta ) ) -> et ) ) ) $=
      ( wa wo wi biimprd adantld pm2.21d adantrd jaao ex ) ABBCIZDEIZJFKARFBSAC
      FBAFCGLMBDFEBDFHNOPQ $.
  $}

  $( A specialized lemma for set theory (to derive the Axiom of Pairing).
     (Contributed by NM, 21-Jun-1993.)  (Proof shortened by Andrew Salmon,
     13-May-2011.)  (Proof shortened by Wolf Lammen, 9-Dec-2012.) $)
  prlem2 $p |- ( ( ( ph /\ ps ) \/ ( ch /\ th ) ) <->
              ( ( ph \/ ch ) /\ ( ( ph /\ ps ) \/ ( ch /\ th ) ) ) ) $=
    ( wa wo simpl orim12i pm4.71ri ) ABEZCDEZFACFJAKCABGCDGHI $.

  ${
    oplem1.1 $e |- ( ph -> ( ps \/ ch ) ) $.
    oplem1.2 $e |- ( ph -> ( th \/ ta ) ) $.
    oplem1.3 $e |- ( ps <-> th ) $.
    oplem1.4 $e |- ( ch -> ( th <-> ta ) ) $.
    $( A specialized lemma for set theory (ordered pair theorem).  (Contributed
       by NM, 18-Oct-1995.)  (Proof shortened by Wolf Lammen, 8-Dec-2012.) $)
    oplem1 $p |- ( ph -> ps ) $=
      ( wn wa notbii ord syl5bir jcad biimpar syl6 pm2.18d sylibr ) ADBADADJZCE
      KDATCETBJACBDHLABCFMNADEGMOCDEIPQRHS $.
  $}

  $( Lemma used in construction of real numbers.  Obsolete as of 1-Jan-2020
     because it no longer has any usages in set.mm and can be replaced by
     theorems such as ~ simpll and ~ jca .  (Contributed by NM, 4-Sep-1995.)
     (Proof shortened by Andrew Salmon, 26-Jun-2011.)
     (New usage is discouraged.)  (Proof modification is discouraged.) $)
  rnlemOLD $p |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) ) <->
  ( ( ( ph /\ ch ) /\ ( ps /\ th ) ) /\ ( ( ph /\ th ) /\ ( ps /\ ch ) ) ) ) $=
    ( wa an4 biimpi an42 biimpri jca adantl impbii ) ABECDEEZACEBDEEZADEBCEEZEM
    NOMNABCDFGOMADBCHZIJOMNOMPGKL $.

  $( A single axiom for Boolean algebra known as DN_1.  See
     ~ http://www-unix.mcs.anl.gov/~~mccune/papers/basax/v12.pdf .
     (Contributed by Jeff Hankins, 3-Jul-2009.)  (Proof shortened by Andrew
     Salmon, 13-May-2011.)  (Proof shortened by Wolf Lammen, 6-Jan-2013.) $)
  dn1 $p |- ( -. ( -. ( -. ( ph \/ ps ) \/ ch ) \/
            -. ( ph \/ -. ( -. ch \/ -. ( ch \/ th ) ) ) ) <-> ch ) $=
    ( wo wn wa wi pm2.45 imnan mpbi biorfi orcom ordir bitri pm4.45 anor orbi2i
    anbi2i 3bitrri ) CABEFZCEZACEZGZUBACFCDEZFEFZEZGUBFUGFEFCCUAAGZEZUDUHCUAAFH
    UHFABIUAAJKLUIUHCEUDCUHMUAACNOOUCUGUBCUFACCUEGUFCDPCUEQORSUBUGQT $.

  $( If a wff is equivalent to its conjunction with another wff, the other wwf
     follows.  (Contributed by Jonathan Ben-Naim, 3-Jun-2011.) $)
  bianir $p |- ( ( ph /\ ( ps <-> ph ) ) -> ps ) $=
    ( wb bicom bi1 impcom sylan2b ) BACAABCZBBADHABABEFG $.

  ${
    jaoi2.1 $e |- ( ( ph \/ ( -. ph /\ ch ) ) -> ps ) $.
    $( Inference removing a negated conjunct in a disjunction of an antecedent
       if this conjunct is part of the disjunction.  (Contributed by Alexander
       van der Vekens, 3-Nov-2017.)  (Proof shortened by Wolf Lammen,
       21-Sep-2018.) $)
    jaoi2 $p |- ( ( ph \/ ch ) -> ps ) $=
      ( wo wn wa pm5.63 sylbi ) ACEAAFCGEBACHDI $.
  $}

  ${
    jaoi3.1 $e |- ( ph -> ps ) $.
    jaoi3.2 $e |- ( ( -. ph /\ ch ) -> ps ) $.
    $( Inference separating a disjunct of an antecedent.  (Contributed by
       Alexander van der Vekens, 25-May-2018.) $)
    jaoi3 $p |- ( ( ph \/ ch ) -> ps ) $=
      ( wn wa jaoi jaoi2 ) ABCABAFCGDEHI $.
  $}

  ${
    cases.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    cases.2 $e |- ( -. ph -> ( ps <-> th ) ) $.
    $( Case disjunction according to the value of ` ph ` .  (Contributed by NM,
       25-Apr-2019.) $)
    cases $p |- ( ps <-> ( ( ph /\ ch ) \/ ( -. ph /\ th ) ) ) $=
      ( wn wo wa exmid biantrur andir pm5.32i orbi12i 3bitri ) BAAGZHZBIABIZPBI
      ZHACIZPDIZHQBAJKAPBLRTSUAABCEMPBDFMNO $.
  $}

  $( Case disjunction according to the value of ` ph ` .  (Contributed by BJ,
     6-Apr-2019.)  (Proof shortened by Wolf Lammen, 2-Jan-2020.) $)
  cases2 $p |- ( ( ( ph /\ ps ) \/ ( -. ph /\ ch ) )
                                   <-> ( ( ph -> ps ) /\ ( -. ph -> ch ) ) ) $=
    ( wa wn wo wi pm3.4 pm2.24 adantr pm2.21 jaoi pm2.27 imdistani orcd adantrr
    jca olcd adantrl pm2.61ian impbii ) ABDZAEZCDZFZABGZUCCGZDZUBUHUDUBUFUGABHA
    UGBACIJQUDUFUGUCUFCABKJUCCHQLAUHUEAUFUEUGAUFDUBUDAUFBABMNOPUCUGUEUFUCUGDUDU
    BUCUGCUCCMNRSTUA $.

  $( Obsolete proof of ~ cases2 as of 2-Jan-2020.  (Contributed by BJ,
     6-Apr-2019.)  (Proof modification is discouraged.)
     (New usage is discouraged.) $)
  cases2OLD $p |- ( ( ( ph /\ ps ) \/ ( -. ph /\ ch ) )
                                   <-> ( ( ph -> ps ) /\ ( -. ph -> ch ) ) ) $=
    ( wa wn wo wi pm3.4 simpl pm2.24 syl jca pm2.21 jaoi exmid pm2.27 imdistani
    orcd ex olcd jaoa ax-mp impbii ) ABDZAEZCDZFZABGZUECGZDZUDUJUFUDUHUIABHUDAU
    IABIACJKLUFUHUIUFUEUHUECIABMKUECHLNAUEFUJUGGAOAUHUGUEUIAUHUGAUHDUDUFAUHBABP
    QRSUEUIUGUEUIDUFUDUEUICUECPQTSUAUBUC $.

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Abbreviated conjunction and disjunction of three wff's
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
$)

  $( Extend wff definition to include 3-way disjunction ('or'). $)
  w3o $a wff ( ph \/ ps \/ ch ) $.
  $( Extend wff definition to include 3-way conjunction ('and'). $)
  w3a $a wff ( ph /\ ps /\ ch ) $.

  $( These abbreviations help eliminate parentheses to aid readability. $)

  $( Define disjunction ('or') of three wff's.  Definition *2.33 of
     [WhiteheadRussell] p. 105.  This abbreviation reduces the number of
     parentheses and emphasizes that the order of bracketing is not important
     by virtue of the associative law ~ orass .  (Contributed by NM,
     8-Apr-1994.) $)
  df-3or $a |- ( ( ph \/ ps \/ ch ) <-> ( ( ph \/ ps ) \/ ch ) ) $.

  $( Define conjunction ('and') of three wff's.  Definition *4.34 of
     [WhiteheadRussell] p. 118.  This abbreviation reduces the number of
     parentheses and emphasizes that the order of bracketing is not important
     by virtue of the associative law ~ anass .  (Contributed by NM,
     8-Apr-1994.) $)
  df-3an $a |- ( ( ph /\ ps /\ ch ) <-> ( ( ph /\ ps ) /\ ch ) ) $.

  $( Associative law for triple disjunction.  (Contributed by NM,
     8-Apr-1994.) $)
  3orass $p |- ( ( ph \/ ps \/ ch ) <-> ( ph \/ ( ps \/ ch ) ) ) $=
    ( w3o wo df-3or orass bitri ) ABCDABECEABCEEABCFABCGH $.

  $( Associative law for triple conjunction.  (Contributed by NM,
     8-Apr-1994.) $)
  3anass $p |- ( ( ph /\ ps /\ ch ) <-> ( ph /\ ( ps /\ ch ) ) ) $=
    ( w3a wa df-3an anass bitri ) ABCDABECEABCEEABCFABCGH $.

  $( Rotation law for triple conjunction.  (Contributed by NM, 8-Apr-1994.) $)
  3anrot $p |- ( ( ph /\ ps /\ ch ) <-> ( ps /\ ch /\ ph ) ) $=
    ( wa w3a ancom 3anass df-3an 3bitr4i ) ABCDZDJADABCEBCAEAJFABCGBCAHI $.

  $( Rotation law for triple disjunction.  (Contributed by NM, 4-Apr-1995.) $)
  3orrot $p |- ( ( ph \/ ps \/ ch ) <-> ( ps \/ ch \/ ph ) ) $=
    ( wo w3o orcom 3orass df-3or 3bitr4i ) ABCDZDJADABCEBCAEAJFABCGBCAHI $.

  $( Commutation law for triple conjunction.  (Contributed by NM,
     21-Apr-1994.) $)
  3ancoma $p |- ( ( ph /\ ps /\ ch ) <-> ( ps /\ ph /\ ch ) ) $=
    ( wa w3a ancom anbi1i df-3an 3bitr4i ) ABDZCDBADZCDABCEBACEJKCABFGABCHBACHI
    $.

  $( Commutation law for triple disjunction.  (Contributed by Mario Carneiro,
     4-Sep-2016.) $)
  3orcoma $p |- ( ( ph \/ ps \/ ch ) <-> ( ps \/ ph \/ ch ) ) $=
    ( wo w3o or12 3orass 3bitr4i ) ABCDDBACDDABCEBACEABCFABCGBACGH $.

  $( Commutation law for triple conjunction.  (Contributed by NM,
     21-Apr-1994.) $)
  3ancomb $p |- ( ( ph /\ ps /\ ch ) <-> ( ph /\ ch /\ ps ) ) $=
    ( w3a 3ancoma 3anrot bitri ) ABCDBACDACBDABCEBACFG $.

  $( Commutation law for triple disjunction.  (Contributed by Scott Fenton,
     20-Apr-2011.) $)
  3orcomb $p |- ( ( ph \/ ps \/ ch ) <-> ( ph \/ ch \/ ps ) ) $=
    ( wo w3o orcom orbi2i 3orass 3bitr4i ) ABCDZDACBDZDABCEACBEJKABCFGABCHACBHI
    $.

  $( Reversal law for triple conjunction.  (Contributed by NM, 21-Apr-1994.) $)
  3anrev $p |- ( ( ph /\ ps /\ ch ) <-> ( ch /\ ps /\ ph ) ) $=
    ( w3a 3ancoma 3anrot bitr4i ) ABCDBACDCBADABCECBAFG $.

  $( Convert triple conjunction to conjunction, then commute.  (Contributed by
     Jonathan Ben-Naim, 3-Jun-2011.) $)
  3anan32 $p |- ( ( ph /\ ps /\ ch ) <-> ( ( ph /\ ch ) /\ ps ) ) $=
    ( w3a wa df-3an an32 bitri ) ABCDABECEACEBEABCFABCGH $.

  $( Convert triple conjunction to conjunction, then commute.  (Contributed by
     Jonathan Ben-Naim, 3-Jun-2011.)  (Proof shortened by Andrew Salmon,
     14-Jun-2011.) $)
  3anan12 $p |- ( ( ph /\ ps /\ ch ) <-> ( ps /\ ( ph /\ ch ) ) ) $=
    ( w3a wa 3ancoma 3anass bitri ) ABCDBACDBACEEABCFBACGH $.

  $( Distribution of triple conjunction over conjunction.  (Contributed by
     David A. Wheeler, 4-Nov-2018.) $)
  anandi3 $p |- ( ( ph /\ ps /\ ch ) <-> ( ( ph /\ ps ) /\ ( ph /\ ch ) ) ) $=
    ( w3a wa 3anass anandi bitri ) ABCDABCEEABEACEEABCFABCGH $.

  $( Distribution of triple conjunction over conjunction.  (Contributed by
     David A. Wheeler, 4-Nov-2018.) $)
  anandi3r $p |- ( ( ph /\ ps /\ ch ) <-> ( ( ph /\ ps ) /\ ( ch /\ ps ) ) ) $=
    ( w3a wa 3anan32 anandir bitri ) ABCDACEBEABECBEEABCFACBGH $.

  $( Triple conjunction expressed in terms of triple disjunction.  (Contributed
     by Jeff Hankins, 15-Aug-2009.) $)
  3anor $p |- ( ( ph /\ ps /\ ch ) <-> -. ( -. ph \/ -. ps \/ -. ch ) ) $=
    ( w3a wa wn w3o df-3an wo anor ianor orbi1i xchbinx df-3or xchbinxr bitri )
    ABCDABEZCEZAFZBFZCFZGZFABCHRSTIZUAIZUBRQFZUAIUDQCJUEUCUAABKLMSTUANOP $.

  $( Negated triple conjunction expressed in terms of triple disjunction.
     (Contributed by Jeff Hankins, 15-Aug-2009.)  (Proof shortened by Andrew
     Salmon, 13-May-2011.) $)
  3ianor $p |- ( -. ( ph /\ ps /\ ch ) <-> ( -. ph \/ -. ps \/ -. ch ) ) $=
    ( wn w3o w3a 3anor con2bii bicomi ) ADBDCDEZABCFZDKJABCGHI $.

  $( Negated triple disjunction as triple conjunction.  (Contributed by Scott
     Fenton, 19-Apr-2011.) $)
  3ioran $p |- ( -. ( ph \/ ps \/ ch ) <-> ( -. ph /\ -. ps /\ -. ch ) ) $=
    ( wo wn wa w3o w3a ioran anbi1i df-3or xchnxbir df-3an 3bitr4i ) ABDZEZCEZF
    ZAEZBEZFZQFABCGZESTQHPUAQABIJOCDRUBOCIABCKLSTQMN $.

  $( Triple disjunction in terms of triple conjunction.  (Contributed by NM,
     8-Oct-2012.) $)
  3oran $p |- ( ( ph \/ ps \/ ch ) <-> -. ( -. ph /\ -. ps /\ -. ch ) ) $=
    ( wn w3a w3o 3ioran con1bii bicomi ) ADBDCDEZDABCFZKJABCGHI $.

  $( Simplification of triple conjunction.  (Contributed by NM,
     21-Apr-1994.) $)
  3simpa $p |- ( ( ph /\ ps /\ ch ) -> ( ph /\ ps ) ) $=
    ( w3a wa df-3an simplbi ) ABCDABECABCFG $.

  $( Simplification of triple conjunction.  (Contributed by NM,
     21-Apr-1994.) $)
  3simpb $p |- ( ( ph /\ ps /\ ch ) -> ( ph /\ ch ) ) $=
    ( w3a wa 3ancomb 3simpa sylbi ) ABCDACBDACEABCFACBGH $.

  $( Simplification of triple conjunction.  (Contributed by NM, 21-Apr-1994.)
     (Proof shortened by Andrew Salmon, 13-May-2011.) $)
  3simpc $p |- ( ( ph /\ ps /\ ch ) -> ( ps /\ ch ) ) $=
    ( w3a wa 3anrot 3simpa sylbi ) ABCDBCADBCEABCFBCAGH $.

  $( Simplification of triple conjunction.  (Contributed by NM,
     21-Apr-1994.) $)
  simp1 $p |- ( ( ph /\ ps /\ ch ) -> ph ) $=
    ( w3a 3simpa simpld ) ABCDABABCEF $.

  $( Simplification of triple conjunction.  (Contributed by NM,
     21-Apr-1994.) $)
  simp2 $p |- ( ( ph /\ ps /\ ch ) -> ps ) $=
    ( w3a 3simpa simprd ) ABCDABABCEF $.

  $( Simplification of triple conjunction.  (Contributed by NM,
     21-Apr-1994.) $)
  simp3 $p |- ( ( ph /\ ps /\ ch ) -> ch ) $=
    ( w3a 3simpc simprd ) ABCDBCABCEF $.

  $( Simplification rule.  (Contributed by Jeff Hankins, 17-Nov-2009.) $)
  simpl1 $p |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ph ) $=
    ( w3a simp1 adantr ) ABCEADABCFG $.

  $( Simplification rule.  (Contributed by Jeff Hankins, 17-Nov-2009.) $)
  simpl2 $p |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ps ) $=
    ( w3a simp2 adantr ) ABCEBDABCFG $.

  $( Simplification rule.  (Contributed by Jeff Hankins, 17-Nov-2009.) $)
  simpl3 $p |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ch ) $=
    ( w3a simp3 adantr ) ABCECDABCFG $.

  $( Simplification rule.  (Contributed by Jeff Hankins, 17-Nov-2009.) $)
  simpr1 $p |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ps ) $=
    ( w3a simp1 adantl ) BCDEBABCDFG $.

  $( Simplification rule.  (Contributed by Jeff Hankins, 17-Nov-2009.) $)
  simpr2 $p |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ch ) $=
    ( w3a simp2 adantl ) BCDECABCDFG $.

  $( Simplification rule.  (Contributed by Jeff Hankins, 17-Nov-2009.) $)
  simpr3 $p |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> th ) $=
    ( w3a simp3 adantl ) BCDEDABCDFG $.

  ${
    3simp1i.1 $e |- ( ph /\ ps /\ ch ) $.
    $( Infer a conjunct from a triple conjunction.  (Contributed by NM,
       19-Apr-2005.) $)
    simp1i $p |- ph $=
      ( w3a simp1 ax-mp ) ABCEADABCFG $.

    $( Infer a conjunct from a triple conjunction.  (Contributed by NM,
       19-Apr-2005.) $)
    simp2i $p |- ps $=
      ( w3a simp2 ax-mp ) ABCEBDABCFG $.

    $( Infer a conjunct from a triple conjunction.  (Contributed by NM,
       19-Apr-2005.) $)
    simp3i $p |- ch $=
      ( w3a simp3 ax-mp ) ABCECDABCFG $.
  $}

  ${
    3simp1d.1 $e |- ( ph -> ( ps /\ ch /\ th ) ) $.
    $( Deduce a conjunct from a triple conjunction.  (Contributed by NM,
       4-Sep-2005.) $)
    simp1d $p |- ( ph -> ps ) $=
      ( w3a simp1 syl ) ABCDFBEBCDGH $.

    $( Deduce a conjunct from a triple conjunction.  (Contributed by NM,
       4-Sep-2005.) $)
    simp2d $p |- ( ph -> ch ) $=
      ( w3a simp2 syl ) ABCDFCEBCDGH $.

    $( Deduce a conjunct from a triple conjunction.  (Contributed by NM,
       4-Sep-2005.) $)
    simp3d $p |- ( ph -> th ) $=
      ( w3a simp3 syl ) ABCDFDEBCDGH $.
  $}

  ${
    3simp1bi.1 $e |- ( ph <-> ( ps /\ ch /\ th ) ) $.
    $( Deduce a conjunct from a triple conjunction.  (Contributed by Jonathan
       Ben-Naim, 3-Jun-2011.) $)
    simp1bi $p |- ( ph -> ps ) $=
      ( w3a biimpi simp1d ) ABCDABCDFEGH $.

    $( Deduce a conjunct from a triple conjunction.  (Contributed by Jonathan
       Ben-Naim, 3-Jun-2011.) $)
    simp2bi $p |- ( ph -> ch ) $=
      ( w3a biimpi simp2d ) ABCDABCDFEGH $.

    $( Deduce a conjunct from a triple conjunction.  (Contributed by Jonathan
       Ben-Naim, 3-Jun-2011.) $)
    simp3bi $p |- ( ph -> th ) $=
      ( w3a biimpi simp3d ) ABCDABCDFEGH $.
  $}

  ${
    3adant.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       16-Jul-1995.) $)
    3adant1 $p |- ( ( th /\ ph /\ ps ) -> ch ) $=
      ( w3a wa 3simpc syl ) DABFABGCDABHEI $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       16-Jul-1995.) $)
    3adant2 $p |- ( ( ph /\ th /\ ps ) -> ch ) $=
      ( w3a wa 3simpb syl ) ADBFABGCADBHEI $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       16-Jul-1995.) $)
    3adant3 $p |- ( ( ph /\ ps /\ th ) -> ch ) $=
      ( w3a wa 3simpa syl ) ABDFABGCABDHEI $.
  $}

  ${
    3ad2ant.1 $e |- ( ph -> ch ) $.
    $( Deduction adding conjuncts to an antecedent.  (Contributed by NM,
       21-Apr-2005.) $)
    3ad2ant1 $p |- ( ( ph /\ ps /\ th ) -> ch ) $=
      ( adantr 3adant2 ) ADCBACDEFG $.

    $( Deduction adding conjuncts to an antecedent.  (Contributed by NM,
       21-Apr-2005.) $)
    3ad2ant2 $p |- ( ( ps /\ ph /\ th ) -> ch ) $=
      ( adantr 3adant1 ) ADCBACDEFG $.

    $( Deduction adding conjuncts to an antecedent.  (Contributed by NM,
       21-Apr-2005.) $)
    3ad2ant3 $p |- ( ( ps /\ th /\ ph ) -> ch ) $=
      ( adantl 3adant1 ) DACBACDEFG $.
  $}

  $( Simplification of triple conjunction.  (Contributed by NM, 9-Nov-2011.) $)
  simp1l $p |- ( ( ( ph /\ ps ) /\ ch /\ th ) -> ph ) $=
    ( wa simpl 3ad2ant1 ) ABECADABFG $.

  $( Simplification of triple conjunction.  (Contributed by NM, 9-Nov-2011.) $)
  simp1r $p |- ( ( ( ph /\ ps ) /\ ch /\ th ) -> ps ) $=
    ( wa simpr 3ad2ant1 ) ABECBDABFG $.

  $( Simplification of triple conjunction.  (Contributed by NM, 9-Nov-2011.) $)
  simp2l $p |- ( ( ph /\ ( ps /\ ch ) /\ th ) -> ps ) $=
    ( wa simpl 3ad2ant2 ) BCEABDBCFG $.

  $( Simplification of triple conjunction.  (Contributed by NM, 9-Nov-2011.) $)
  simp2r $p |- ( ( ph /\ ( ps /\ ch ) /\ th ) -> ch ) $=
    ( wa simpr 3ad2ant2 ) BCEACDBCFG $.

  $( Simplification of triple conjunction.  (Contributed by NM, 9-Nov-2011.) $)
  simp3l $p |- ( ( ph /\ ps /\ ( ch /\ th ) ) -> ch ) $=
    ( wa simpl 3ad2ant3 ) CDEACBCDFG $.

  $( Simplification of triple conjunction.  (Contributed by NM, 9-Nov-2011.) $)
  simp3r $p |- ( ( ph /\ ps /\ ( ch /\ th ) ) -> th ) $=
    ( wa simpr 3ad2ant3 ) CDEADBCDFG $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp11 $p |- ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) -> ph ) $=
    ( w3a simp1 3ad2ant1 ) ABCFDAEABCGH $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp12 $p |- ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) -> ps ) $=
    ( w3a simp2 3ad2ant1 ) ABCFDBEABCGH $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp13 $p |- ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) -> ch ) $=
    ( w3a simp3 3ad2ant1 ) ABCFDCEABCGH $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp21 $p |- ( ( ph /\ ( ps /\ ch /\ th ) /\ ta ) -> ps ) $=
    ( w3a simp1 3ad2ant2 ) BCDFABEBCDGH $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp22 $p |- ( ( ph /\ ( ps /\ ch /\ th ) /\ ta ) -> ch ) $=
    ( w3a simp2 3ad2ant2 ) BCDFACEBCDGH $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp23 $p |- ( ( ph /\ ( ps /\ ch /\ th ) /\ ta ) -> th ) $=
    ( w3a simp3 3ad2ant2 ) BCDFADEBCDGH $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp31 $p |- ( ( ph /\ ps /\ ( ch /\ th /\ ta ) ) -> ch ) $=
    ( w3a simp1 3ad2ant3 ) CDEFACBCDEGH $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp32 $p |- ( ( ph /\ ps /\ ( ch /\ th /\ ta ) ) -> th ) $=
    ( w3a simp2 3ad2ant3 ) CDEFADBCDEGH $.

  $( Simplification of doubly triple conjunction.  (Contributed by NM,
     17-Nov-2011.) $)
  simp33 $p |- ( ( ph /\ ps /\ ( ch /\ th /\ ta ) ) -> ta ) $=
    ( w3a simp3 3ad2ant3 ) CDEFAEBCDEGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpll1 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th ) /\ ta ) -> ph ) $=
    ( w3a wa simpl1 adantr ) ABCFDGAEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpll2 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th ) /\ ta ) -> ps ) $=
    ( w3a wa simpl2 adantr ) ABCFDGBEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpll3 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th ) /\ ta ) -> ch ) $=
    ( w3a wa simpl3 adantr ) ABCFDGCEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simplr1 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) ) /\ ta ) -> ph ) $=
    ( w3a wa simpr1 adantr ) DABCFGAEDABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simplr2 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) ) /\ ta ) -> ps ) $=
    ( w3a wa simpr2 adantr ) DABCFGBEDABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simplr3 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) ) /\ ta ) -> ch ) $=
    ( w3a wa simpr3 adantr ) DABCFGCEDABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simprl1 $p |- ( ( ta /\ ( ( ph /\ ps /\ ch ) /\ th ) ) -> ph ) $=
    ( w3a wa simpl1 adantl ) ABCFDGAEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simprl2 $p |- ( ( ta /\ ( ( ph /\ ps /\ ch ) /\ th ) ) -> ps ) $=
    ( w3a wa simpl2 adantl ) ABCFDGBEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simprl3 $p |- ( ( ta /\ ( ( ph /\ ps /\ ch ) /\ th ) ) -> ch ) $=
    ( w3a wa simpl3 adantl ) ABCFDGCEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simprr1 $p |- ( ( ta /\ ( th /\ ( ph /\ ps /\ ch ) ) ) -> ph ) $=
    ( w3a wa simpr1 adantl ) DABCFGAEDABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simprr2 $p |- ( ( ta /\ ( th /\ ( ph /\ ps /\ ch ) ) ) -> ps ) $=
    ( w3a wa simpr2 adantl ) DABCFGBEDABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simprr3 $p |- ( ( ta /\ ( th /\ ( ph /\ ps /\ ch ) ) ) -> ch ) $=
    ( w3a wa simpr3 adantl ) DABCFGCEDABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl1l $p |- ( ( ( ( ph /\ ps ) /\ ch /\ th ) /\ ta ) -> ph ) $=
    ( wa w3a simp1l adantr ) ABFCDGAEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl1r $p |- ( ( ( ( ph /\ ps ) /\ ch /\ th ) /\ ta ) -> ps ) $=
    ( wa w3a simp1r adantr ) ABFCDGBEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl2l $p |- ( ( ( ch /\ ( ph /\ ps ) /\ th ) /\ ta ) -> ph ) $=
    ( wa w3a simp2l adantr ) CABFDGAECABDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl2r $p |- ( ( ( ch /\ ( ph /\ ps ) /\ th ) /\ ta ) -> ps ) $=
    ( wa w3a simp2r adantr ) CABFDGBECABDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl3l $p |- ( ( ( ch /\ th /\ ( ph /\ ps ) ) /\ ta ) -> ph ) $=
    ( wa w3a simp3l adantr ) CDABFGAECDABHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl3r $p |- ( ( ( ch /\ th /\ ( ph /\ ps ) ) /\ ta ) -> ps ) $=
    ( wa w3a simp3r adantr ) CDABFGBECDABHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr1l $p |- ( ( ta /\ ( ( ph /\ ps ) /\ ch /\ th ) ) -> ph ) $=
    ( wa w3a simp1l adantl ) ABFCDGAEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr1r $p |- ( ( ta /\ ( ( ph /\ ps ) /\ ch /\ th ) ) -> ps ) $=
    ( wa w3a simp1r adantl ) ABFCDGBEABCDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr2l $p |- ( ( ta /\ ( ch /\ ( ph /\ ps ) /\ th ) ) -> ph ) $=
    ( wa w3a simp2l adantl ) CABFDGAECABDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr2r $p |- ( ( ta /\ ( ch /\ ( ph /\ ps ) /\ th ) ) -> ps ) $=
    ( wa w3a simp2r adantl ) CABFDGBECABDHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr3l $p |- ( ( ta /\ ( ch /\ th /\ ( ph /\ ps ) ) ) -> ph ) $=
    ( wa w3a simp3l adantl ) CDABFGAECDABHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr3r $p |- ( ( ta /\ ( ch /\ th /\ ( ph /\ ps ) ) ) -> ps ) $=
    ( wa w3a simp3r adantl ) CDABFGBECDABHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1ll $p |- ( ( ( ( ph /\ ps ) /\ ch ) /\ th /\ ta ) -> ph ) $=
    ( wa simpll 3ad2ant1 ) ABFCFDAEABCGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1lr $p |- ( ( ( ( ph /\ ps ) /\ ch ) /\ th /\ ta ) -> ps ) $=
    ( wa simplr 3ad2ant1 ) ABFCFDBEABCGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1rl $p |- ( ( ( ch /\ ( ph /\ ps ) ) /\ th /\ ta ) -> ph ) $=
    ( wa simprl 3ad2ant1 ) CABFFDAECABGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1rr $p |- ( ( ( ch /\ ( ph /\ ps ) ) /\ th /\ ta ) -> ps ) $=
    ( wa simprr 3ad2ant1 ) CABFFDBECABGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2ll $p |- ( ( th /\ ( ( ph /\ ps ) /\ ch ) /\ ta ) -> ph ) $=
    ( wa simpll 3ad2ant2 ) ABFCFDAEABCGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2lr $p |- ( ( th /\ ( ( ph /\ ps ) /\ ch ) /\ ta ) -> ps ) $=
    ( wa simplr 3ad2ant2 ) ABFCFDBEABCGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2rl $p |- ( ( th /\ ( ch /\ ( ph /\ ps ) ) /\ ta ) -> ph ) $=
    ( wa simprl 3ad2ant2 ) CABFFDAECABGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2rr $p |- ( ( th /\ ( ch /\ ( ph /\ ps ) ) /\ ta ) -> ps ) $=
    ( wa simprr 3ad2ant2 ) CABFFDBECABGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3ll $p |- ( ( th /\ ta /\ ( ( ph /\ ps ) /\ ch ) ) -> ph ) $=
    ( wa simpll 3ad2ant3 ) ABFCFDAEABCGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3lr $p |- ( ( th /\ ta /\ ( ( ph /\ ps ) /\ ch ) ) -> ps ) $=
    ( wa simplr 3ad2ant3 ) ABFCFDBEABCGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3rl $p |- ( ( th /\ ta /\ ( ch /\ ( ph /\ ps ) ) ) -> ph ) $=
    ( wa simprl 3ad2ant3 ) CABFFDAECABGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3rr $p |- ( ( th /\ ta /\ ( ch /\ ( ph /\ ps ) ) ) -> ps ) $=
    ( wa simprr 3ad2ant3 ) CABFFDBECABGH $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl11 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ et ) -> ph ) $=
    ( w3a simp11 adantr ) ABCGDEGAFABCDEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl12 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ et ) -> ps ) $=
    ( w3a simp12 adantr ) ABCGDEGBFABCDEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl13 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ et ) -> ch ) $=
    ( w3a simp13 adantr ) ABCGDEGCFABCDEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl21 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ et ) -> ph ) $=
    ( w3a simp21 adantr ) DABCGEGAFDABCEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl22 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ et ) -> ps ) $=
    ( w3a simp22 adantr ) DABCGEGBFDABCEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl23 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ et ) -> ch ) $=
    ( w3a simp23 adantr ) DABCGEGCFDABCEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl31 $p |- ( ( ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ et ) -> ph ) $=
    ( w3a simp31 adantr ) DEABCGGAFDEABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl32 $p |- ( ( ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ et ) -> ps ) $=
    ( w3a simp32 adantr ) DEABCGGBFDEABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpl33 $p |- ( ( ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ et ) -> ch ) $=
    ( w3a simp33 adantr ) DEABCGGCFDEABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr11 $p |- ( ( et /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) ) -> ph ) $=
    ( w3a simp11 adantl ) ABCGDEGAFABCDEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr12 $p |- ( ( et /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) ) -> ps ) $=
    ( w3a simp12 adantl ) ABCGDEGBFABCDEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr13 $p |- ( ( et /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) ) -> ch ) $=
    ( w3a simp13 adantl ) ABCGDEGCFABCDEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr21 $p |- ( ( et /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) ) -> ph ) $=
    ( w3a simp21 adantl ) DABCGEGAFDABCEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr22 $p |- ( ( et /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) ) -> ps ) $=
    ( w3a simp22 adantl ) DABCGEGBFDABCEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr23 $p |- ( ( et /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) ) -> ch ) $=
    ( w3a simp23 adantl ) DABCGEGCFDABCEHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr31 $p |- ( ( et /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) ) -> ph ) $=
    ( w3a simp31 adantl ) DEABCGGAFDEABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr32 $p |- ( ( et /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) ) -> ps ) $=
    ( w3a simp32 adantl ) DEABCGGBFDEABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simpr33 $p |- ( ( et /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) ) -> ch ) $=
    ( w3a simp33 adantl ) DEABCGGCFDEABCHI $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1l1 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th ) /\ ta /\ et ) -> ph ) $=
    ( w3a wa simpl1 3ad2ant1 ) ABCGDHEAFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1l2 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th ) /\ ta /\ et ) -> ps ) $=
    ( w3a wa simpl2 3ad2ant1 ) ABCGDHEBFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1l3 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th ) /\ ta /\ et ) -> ch ) $=
    ( w3a wa simpl3 3ad2ant1 ) ABCGDHECFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1r1 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) ) /\ ta /\ et ) -> ph ) $=
    ( w3a wa simpr1 3ad2ant1 ) DABCGHEAFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1r2 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) ) /\ ta /\ et ) -> ps ) $=
    ( w3a wa simpr2 3ad2ant1 ) DABCGHEBFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp1r3 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) ) /\ ta /\ et ) -> ch ) $=
    ( w3a wa simpr3 3ad2ant1 ) DABCGHECFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2l1 $p |- ( ( ta /\ ( ( ph /\ ps /\ ch ) /\ th ) /\ et ) -> ph ) $=
    ( w3a wa simpl1 3ad2ant2 ) ABCGDHEAFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2l2 $p |- ( ( ta /\ ( ( ph /\ ps /\ ch ) /\ th ) /\ et ) -> ps ) $=
    ( w3a wa simpl2 3ad2ant2 ) ABCGDHEBFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2l3 $p |- ( ( ta /\ ( ( ph /\ ps /\ ch ) /\ th ) /\ et ) -> ch ) $=
    ( w3a wa simpl3 3ad2ant2 ) ABCGDHECFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2r1 $p |- ( ( ta /\ ( th /\ ( ph /\ ps /\ ch ) ) /\ et ) -> ph ) $=
    ( w3a wa simpr1 3ad2ant2 ) DABCGHEAFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2r2 $p |- ( ( ta /\ ( th /\ ( ph /\ ps /\ ch ) ) /\ et ) -> ps ) $=
    ( w3a wa simpr2 3ad2ant2 ) DABCGHEBFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp2r3 $p |- ( ( ta /\ ( th /\ ( ph /\ ps /\ ch ) ) /\ et ) -> ch ) $=
    ( w3a wa simpr3 3ad2ant2 ) DABCGHECFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3l1 $p |- ( ( ta /\ et /\ ( ( ph /\ ps /\ ch ) /\ th ) ) -> ph ) $=
    ( w3a wa simpl1 3ad2ant3 ) ABCGDHEAFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3l2 $p |- ( ( ta /\ et /\ ( ( ph /\ ps /\ ch ) /\ th ) ) -> ps ) $=
    ( w3a wa simpl2 3ad2ant3 ) ABCGDHEBFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3l3 $p |- ( ( ta /\ et /\ ( ( ph /\ ps /\ ch ) /\ th ) ) -> ch ) $=
    ( w3a wa simpl3 3ad2ant3 ) ABCGDHECFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3r1 $p |- ( ( ta /\ et /\ ( th /\ ( ph /\ ps /\ ch ) ) ) -> ph ) $=
    ( w3a wa simpr1 3ad2ant3 ) DABCGHEAFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3r2 $p |- ( ( ta /\ et /\ ( th /\ ( ph /\ ps /\ ch ) ) ) -> ps ) $=
    ( w3a wa simpr2 3ad2ant3 ) DABCGHEBFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp3r3 $p |- ( ( ta /\ et /\ ( th /\ ( ph /\ ps /\ ch ) ) ) -> ch ) $=
    ( w3a wa simpr3 3ad2ant3 ) DABCGHECFDABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp11l $p |- ( ( ( ( ph /\ ps ) /\ ch /\ th ) /\ ta /\ et ) -> ph ) $=
    ( wa w3a simp1l 3ad2ant1 ) ABGCDHEAFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp11r $p |- ( ( ( ( ph /\ ps ) /\ ch /\ th ) /\ ta /\ et ) -> ps ) $=
    ( wa w3a simp1r 3ad2ant1 ) ABGCDHEBFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp12l $p |- ( ( ( ch /\ ( ph /\ ps ) /\ th ) /\ ta /\ et ) -> ph ) $=
    ( wa w3a simp2l 3ad2ant1 ) CABGDHEAFCABDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp12r $p |- ( ( ( ch /\ ( ph /\ ps ) /\ th ) /\ ta /\ et ) -> ps ) $=
    ( wa w3a simp2r 3ad2ant1 ) CABGDHEBFCABDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp13l $p |- ( ( ( ch /\ th /\ ( ph /\ ps ) ) /\ ta /\ et ) -> ph ) $=
    ( wa w3a simp3l 3ad2ant1 ) CDABGHEAFCDABIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp13r $p |- ( ( ( ch /\ th /\ ( ph /\ ps ) ) /\ ta /\ et ) -> ps ) $=
    ( wa w3a simp3r 3ad2ant1 ) CDABGHEBFCDABIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp21l $p |- ( ( ta /\ ( ( ph /\ ps ) /\ ch /\ th ) /\ et ) -> ph ) $=
    ( wa w3a simp1l 3ad2ant2 ) ABGCDHEAFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp21r $p |- ( ( ta /\ ( ( ph /\ ps ) /\ ch /\ th ) /\ et ) -> ps ) $=
    ( wa w3a simp1r 3ad2ant2 ) ABGCDHEBFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp22l $p |- ( ( ta /\ ( ch /\ ( ph /\ ps ) /\ th ) /\ et ) -> ph ) $=
    ( wa w3a simp2l 3ad2ant2 ) CABGDHEAFCABDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp22r $p |- ( ( ta /\ ( ch /\ ( ph /\ ps ) /\ th ) /\ et ) -> ps ) $=
    ( wa w3a simp2r 3ad2ant2 ) CABGDHEBFCABDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp23l $p |- ( ( ta /\ ( ch /\ th /\ ( ph /\ ps ) ) /\ et ) -> ph ) $=
    ( wa w3a simp3l 3ad2ant2 ) CDABGHEAFCDABIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp23r $p |- ( ( ta /\ ( ch /\ th /\ ( ph /\ ps ) ) /\ et ) -> ps ) $=
    ( wa w3a simp3r 3ad2ant2 ) CDABGHEBFCDABIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp31l $p |- ( ( ta /\ et /\ ( ( ph /\ ps ) /\ ch /\ th ) ) -> ph ) $=
    ( wa w3a simp1l 3ad2ant3 ) ABGCDHEAFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp31r $p |- ( ( ta /\ et /\ ( ( ph /\ ps ) /\ ch /\ th ) ) -> ps ) $=
    ( wa w3a simp1r 3ad2ant3 ) ABGCDHEBFABCDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp32l $p |- ( ( ta /\ et /\ ( ch /\ ( ph /\ ps ) /\ th ) ) -> ph ) $=
    ( wa w3a simp2l 3ad2ant3 ) CABGDHEAFCABDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp32r $p |- ( ( ta /\ et /\ ( ch /\ ( ph /\ ps ) /\ th ) ) -> ps ) $=
    ( wa w3a simp2r 3ad2ant3 ) CABGDHEBFCABDIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp33l $p |- ( ( ta /\ et /\ ( ch /\ th /\ ( ph /\ ps ) ) ) -> ph ) $=
    ( wa w3a simp3l 3ad2ant3 ) CDABGHEAFCDABIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp33r $p |- ( ( ta /\ et /\ ( ch /\ th /\ ( ph /\ ps ) ) ) -> ps ) $=
    ( wa w3a simp3r 3ad2ant3 ) CDABGHEBFCDABIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp111 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ et /\ ze ) -> ph ) $=
    ( w3a simp11 3ad2ant1 ) ABCHDEHFAGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp112 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ et /\ ze ) -> ps ) $=
    ( w3a simp12 3ad2ant1 ) ABCHDEHFBGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp113 $p |- ( ( ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ et /\ ze ) -> ch ) $=
    ( w3a simp13 3ad2ant1 ) ABCHDEHFCGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp121 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ et /\ ze ) -> ph ) $=
    ( w3a simp21 3ad2ant1 ) DABCHEHFAGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp122 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ et /\ ze ) -> ps ) $=
    ( w3a simp22 3ad2ant1 ) DABCHEHFBGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp123 $p |- ( ( ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ et /\ ze ) -> ch ) $=
    ( w3a simp23 3ad2ant1 ) DABCHEHFCGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp131 $p |- ( ( ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ et /\ ze ) -> ph ) $=
    ( w3a simp31 3ad2ant1 ) DEABCHHFAGDEABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp132 $p |- ( ( ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ et /\ ze ) -> ps ) $=
    ( w3a simp32 3ad2ant1 ) DEABCHHFBGDEABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp133 $p |- ( ( ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ et /\ ze ) -> ch ) $=
    ( w3a simp33 3ad2ant1 ) DEABCHHFCGDEABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp211 $p |- ( ( et /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ ze ) -> ph ) $=
    ( w3a simp11 3ad2ant2 ) ABCHDEHFAGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp212 $p |- ( ( et /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ ze ) -> ps ) $=
    ( w3a simp12 3ad2ant2 ) ABCHDEHFBGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp213 $p |- ( ( et /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) /\ ze ) -> ch ) $=
    ( w3a simp13 3ad2ant2 ) ABCHDEHFCGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp221 $p |- ( ( et /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ ze ) -> ph ) $=
    ( w3a simp21 3ad2ant2 ) DABCHEHFAGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp222 $p |- ( ( et /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ ze ) -> ps ) $=
    ( w3a simp22 3ad2ant2 ) DABCHEHFBGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp223 $p |- ( ( et /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) /\ ze ) -> ch ) $=
    ( w3a simp23 3ad2ant2 ) DABCHEHFCGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp231 $p |- ( ( et /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ ze ) -> ph ) $=
    ( w3a simp31 3ad2ant2 ) DEABCHHFAGDEABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp232 $p |- ( ( et /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ ze ) -> ps ) $=
    ( w3a simp32 3ad2ant2 ) DEABCHHFBGDEABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp233 $p |- ( ( et /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) /\ ze ) -> ch ) $=
    ( w3a simp33 3ad2ant2 ) DEABCHHFCGDEABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp311 $p |- ( ( et /\ ze /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) ) -> ph ) $=
    ( w3a simp11 3ad2ant3 ) ABCHDEHFAGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp312 $p |- ( ( et /\ ze /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) ) -> ps ) $=
    ( w3a simp12 3ad2ant3 ) ABCHDEHFBGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp313 $p |- ( ( et /\ ze /\ ( ( ph /\ ps /\ ch ) /\ th /\ ta ) ) -> ch ) $=
    ( w3a simp13 3ad2ant3 ) ABCHDEHFCGABCDEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp321 $p |- ( ( et /\ ze /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) ) -> ph ) $=
    ( w3a simp21 3ad2ant3 ) DABCHEHFAGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp322 $p |- ( ( et /\ ze /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) ) -> ps ) $=
    ( w3a simp22 3ad2ant3 ) DABCHEHFBGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp323 $p |- ( ( et /\ ze /\ ( th /\ ( ph /\ ps /\ ch ) /\ ta ) ) -> ch ) $=
    ( w3a simp23 3ad2ant3 ) DABCHEHFCGDABCEIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp331 $p |- ( ( et /\ ze /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) ) -> ph ) $=
    ( w3a simp31 3ad2ant3 ) DEABCHHFAGDEABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp332 $p |- ( ( et /\ ze /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) ) -> ps ) $=
    ( w3a simp32 3ad2ant3 ) DEABCHHFBGDEABCIJ $.

  $( Simplification of conjunction.  (Contributed by NM, 9-Mar-2012.) $)
  simp333 $p |- ( ( et /\ ze /\ ( th /\ ta /\ ( ph /\ ps /\ ch ) ) ) -> ch ) $=
    ( w3a simp33 3ad2ant3 ) DEABCHHFCGDEABCIJ $.

  ${
    3adantl.1 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       24-Feb-2005.) $)
    3adantl1 $p |- ( ( ( ta /\ ph /\ ps ) /\ ch ) -> th ) $=
      ( w3a wa 3simpc sylan ) EABGABHCDEABIFJ $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       24-Feb-2005.) $)
    3adantl2 $p |- ( ( ( ph /\ ta /\ ps ) /\ ch ) -> th ) $=
      ( w3a wa 3simpb sylan ) AEBGABHCDAEBIFJ $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       24-Feb-2005.) $)
    3adantl3 $p |- ( ( ( ph /\ ps /\ ta ) /\ ch ) -> th ) $=
      ( w3a wa 3simpa sylan ) ABEGABHCDABEIFJ $.
  $}

  ${
    3adantr.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       27-Apr-2005.) $)
    3adantr1 $p |- ( ( ph /\ ( ta /\ ps /\ ch ) ) -> th ) $=
      ( w3a wa 3simpc sylan2 ) EBCGABCHDEBCIFJ $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       27-Apr-2005.) $)
    3adantr2 $p |- ( ( ph /\ ( ps /\ ta /\ ch ) ) -> th ) $=
      ( w3a wa 3simpb sylan2 ) BECGABCHDBECIFJ $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       27-Apr-2005.) $)
    3adantr3 $p |- ( ( ph /\ ( ps /\ ch /\ ta ) ) -> th ) $=
      ( w3a wa 3simpa sylan2 ) BCEGABCHDBCEIFJ $.
  $}

  ${
    3ad2antl.1 $e |- ( ( ph /\ ch ) -> th ) $.
    $( Deduction adding conjuncts to antecedent.  (Contributed by NM,
       4-Aug-2007.) $)
    3ad2antl1 $p |- ( ( ( ph /\ ps /\ ta ) /\ ch ) -> th ) $=
      ( adantlr 3adantl2 ) AECDBACDEFGH $.

    $( Deduction adding conjuncts to antecedent.  (Contributed by NM,
       4-Aug-2007.) $)
    3ad2antl2 $p |- ( ( ( ps /\ ph /\ ta ) /\ ch ) -> th ) $=
      ( adantlr 3adantl1 ) AECDBACDEFGH $.

    $( Deduction adding conjuncts to antecedent.  (Contributed by NM,
       4-Aug-2007.) $)
    3ad2antl3 $p |- ( ( ( ps /\ ta /\ ph ) /\ ch ) -> th ) $=
      ( adantll 3adantl1 ) EACDBACDEFGH $.

    $( Deduction adding conjuncts to antecedent.  (Contributed by NM,
       25-Dec-2007.) $)
    3ad2antr1 $p |- ( ( ph /\ ( ch /\ ps /\ ta ) ) -> th ) $=
      ( adantrr 3adantr3 ) ACBDEACDBFGH $.

    $( Deduction adding conjuncts to antecedent.  (Contributed by NM,
       27-Dec-2007.) $)
    3ad2antr2 $p |- ( ( ph /\ ( ps /\ ch /\ ta ) ) -> th ) $=
      ( adantrl 3adantr3 ) ABCDEACDBFGH $.

    $( Deduction adding conjuncts to antecedent.  (Contributed by NM,
       30-Dec-2007.) $)
    3ad2antr3 $p |- ( ( ph /\ ( ps /\ ta /\ ch ) ) -> th ) $=
      ( adantrl 3adantr1 ) AECDBACDEFGH $.
  $}

  ${
    3anibar.1 $e |- ( ( ph /\ ps /\ ch ) -> ( th <-> ( ch /\ ta ) ) ) $.
    $( Remove a hypothesis from the second member of a biimplication.
       (Contributed by FL, 22-Jul-2008.) $)
    3anibar $p |- ( ( ph /\ ps /\ ch ) -> ( th <-> ta ) ) $=
      ( w3a wa simp3 biantrurd bitr4d ) ABCGZDCEHEFLCEABCIJK $.
  $}

  $( Introduction in triple disjunction.  (Contributed by NM, 4-Apr-1995.) $)
  3mix1 $p |- ( ph -> ( ph \/ ps \/ ch ) ) $=
    ( wo w3o orc 3orass sylibr ) AABCDZDABCEAIFABCGH $.

  $( Introduction in triple disjunction.  (Contributed by NM, 4-Apr-1995.) $)
  3mix2 $p |- ( ph -> ( ps \/ ph \/ ch ) ) $=
    ( w3o 3mix1 3orrot sylibr ) AACBDBACDACBEBACFG $.

  $( Introduction in triple disjunction.  (Contributed by NM, 4-Apr-1995.) $)
  3mix3 $p |- ( ph -> ( ps \/ ch \/ ph ) ) $=
    ( w3o 3mix1 3orrot sylib ) AABCDBCADABCEABCFG $.

  ${
    3mixi.1 $e |- ph $.
    $( Introduction in triple disjunction.  (Contributed by Mario Carneiro,
       6-Oct-2014.) $)
    3mix1i $p |- ( ph \/ ps \/ ch ) $=
      ( w3o 3mix1 ax-mp ) AABCEDABCFG $.

    $( Introduction in triple disjunction.  (Contributed by Mario Carneiro,
       6-Oct-2014.) $)
    3mix2i $p |- ( ps \/ ph \/ ch ) $=
      ( w3o 3mix2 ax-mp ) ABACEDABCFG $.

    $( Introduction in triple disjunction.  (Contributed by Mario Carneiro,
       6-Oct-2014.) $)
    3mix3i $p |- ( ps \/ ch \/ ph ) $=
      ( w3o 3mix3 ax-mp ) ABCAEDABCFG $.
  $}

  ${
    3mixd.1 $e |- ( ph -> ps ) $.
    $( Deduction introducing triple disjunction.  (Contributed by Scott Fenton,
       8-Jun-2011.) $)
    3mix1d $p |- ( ph -> ( ps \/ ch \/ th ) ) $=
      ( w3o 3mix1 syl ) ABBCDFEBCDGH $.

    $( Deduction introducing triple disjunction.  (Contributed by Scott Fenton,
       8-Jun-2011.) $)
    3mix2d $p |- ( ph -> ( ch \/ ps \/ th ) ) $=
      ( w3o 3mix2 syl ) ABCBDFEBCDGH $.

    $( Deduction introducing triple disjunction.  (Contributed by Scott Fenton,
       8-Jun-2011.) $)
    3mix3d $p |- ( ph -> ( ch \/ th \/ ps ) ) $=
      ( w3o 3mix3 syl ) ABCDBFEBCDGH $.
  $}

  ${
    3pm3.2i.1 $e |- ph $.
    3pm3.2i.2 $e |- ps $.
    3pm3.2i.3 $e |- ch $.
    $( Infer conjunction of premises.  (Contributed by NM, 10-Feb-1995.) $)
    3pm3.2i $p |- ( ph /\ ps /\ ch ) $=
      ( w3a wa pm3.2i df-3an mpbir2an ) ABCGABHCABDEIFABCJK $.
  $}

  ${
    $( Version of ~ pm3.2 for a triple conjunction.  (Contributed by Alan Sare,
       24-Oct-2011.) $)
    pm3.2an3 $p |- ( ph -> ( ps -> ( ch -> ( ph /\ ps /\ ch ) ) ) ) $=
      ( wa w3a wi pm3.2 ex df-3an bicomi syl8ib ) ABCABDZCDZABCEZABCMFLCGHNMABC
      IJK $.
  $}

  ${
    3jca.1 $e |- ( ph -> ps ) $.
    3jca.2 $e |- ( ph -> ch ) $.
    3jca.3 $e |- ( ph -> th ) $.
    $( Join consequents with conjunction.  (Contributed by NM, 9-Apr-1994.) $)
    3jca $p |- ( ph -> ( ps /\ ch /\ th ) ) $=
      ( wa w3a jca31 df-3an sylibr ) ABCHDHBCDIABCDEFGJBCDKL $.
  $}

  ${
    3jcad.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3jcad.2 $e |- ( ph -> ( ps -> th ) ) $.
    3jcad.3 $e |- ( ph -> ( ps -> ta ) ) $.
    $( Deduction conjoining the consequents of three implications.
       (Contributed by NM, 25-Sep-2005.) $)
    3jcad $p |- ( ph -> ( ps -> ( ch /\ th /\ ta ) ) ) $=
      ( w3a wa imp 3jca ex ) ABCDEIABJCDEABCFKABDGKABEHKLM $.
  $}

  ${
    mpbir3an.1 $e |- ps $.
    mpbir3an.2 $e |- ch $.
    mpbir3an.3 $e |- th $.
    mpbir3an.4 $e |- ( ph <-> ( ps /\ ch /\ th ) ) $.
    $( Detach a conjunction of truths in a biconditional.  (Contributed by NM,
       16-Sep-2011.) $)
    mpbir3an $p |- ph $=
      ( w3a 3pm3.2i mpbir ) ABCDIBCDEFGJHK $.
  $}

  ${
    mpbir3and.1 $e |- ( ph -> ch ) $.
    mpbir3and.2 $e |- ( ph -> th ) $.
    mpbir3and.3 $e |- ( ph -> ta ) $.
    mpbir3and.4 $e |- ( ph -> ( ps <-> ( ch /\ th /\ ta ) ) ) $.
    $( Detach a conjunction of truths in a biconditional.  (Contributed by
       Mario Carneiro, 11-May-2014.)  (Revised by Mario Carneiro,
       9-Jan-2015.) $)
    mpbir3and $p |- ( ph -> ps ) $=
      ( w3a 3jca mpbird ) ABCDEJACDEFGHKIL $.
  $}

  ${
    syl3anbrc.1 $e |- ( ph -> ps ) $.
    syl3anbrc.2 $e |- ( ph -> ch ) $.
    syl3anbrc.3 $e |- ( ph -> th ) $.
    syl3anbrc.4 $e |- ( ta <-> ( ps /\ ch /\ th ) ) $.
    $( Syllogism inference.  (Contributed by Mario Carneiro, 11-May-2014.) $)
    syl3anbrc $p |- ( ph -> ta ) $=
      ( w3a 3jca sylibr ) ABCDJEABCDFGHKIL $.
  $}

  ${
    3anim123i.1 $e |- ( ph -> ps ) $.
    3anim123i.2 $e |- ( ch -> th ) $.
    3anim123i.3 $e |- ( ta -> et ) $.
    $( Join antecedents and consequents with conjunction.  (Contributed by NM,
       8-Apr-1994.) $)
    3anim123i $p |- ( ( ph /\ ch /\ ta ) -> ( ps /\ th /\ et ) ) $=
      ( w3a 3ad2ant1 3ad2ant2 3ad2ant3 3jca ) ACEJBDFACBEGKCADEHLEAFCIMN $.
  $}

  ${
    3animi.1 $e |- ( ph -> ps ) $.
    $( Add two conjuncts to antecedent and consequent.  (Contributed by Jeff
       Hankins, 16-Aug-2009.) $)
    3anim1i $p |- ( ( ph /\ ch /\ th ) -> ( ps /\ ch /\ th ) ) $=
      ( id 3anim123i ) ABCCDDECFDFG $.

    $( Add two conjuncts to antecedent and consequent.  (Contributed by AV,
       21-Nov-2019.) $)
    3anim2i $p |- ( ( ch /\ ph /\ th ) -> ( ch /\ ps /\ th ) ) $=
      ( id 3anim123i ) CCABDDCFEDFG $.

    $( Add two conjuncts to antecedent and consequent.  (Contributed by Jeff
       Hankins, 19-Aug-2009.) $)
    3anim3i $p |- ( ( ch /\ th /\ ph ) -> ( ch /\ th /\ ps ) ) $=
      ( id 3anim123i ) CCDDABCFDFEG $.
  $}

  ${
    bi3.1 $e |- ( ph <-> ps ) $.
    bi3.2 $e |- ( ch <-> th ) $.
    bi3.3 $e |- ( ta <-> et ) $.
    $( Join 3 biconditionals with conjunction.  (Contributed by NM,
       21-Apr-1994.) $)
    3anbi123i $p |- ( ( ph /\ ch /\ ta ) <-> ( ps /\ th /\ et ) ) $=
      ( wa w3a anbi12i df-3an 3bitr4i ) ACJZEJBDJZFJACEKBDFKOPEFABCDGHLILACEMBD
      FMN $.

    $( Join 3 biconditionals with disjunction.  (Contributed by NM,
       17-May-1994.) $)
    3orbi123i $p |- ( ( ph \/ ch \/ ta ) <-> ( ps \/ th \/ et ) ) $=
      ( wo w3o orbi12i df-3or 3bitr4i ) ACJZEJBDJZFJACEKBDFKOPEFABCDGHLILACEMBD
      FMN $.
  $}

  ${
    3anbi1i.1 $e |- ( ph <-> ps ) $.
    $( Inference adding two conjuncts to each side of a biconditional.
       (Contributed by NM, 8-Sep-2006.) $)
    3anbi1i $p |- ( ( ph /\ ch /\ th ) <-> ( ps /\ ch /\ th ) ) $=
      ( biid 3anbi123i ) ABCCDDECFDFG $.

    $( Inference adding two conjuncts to each side of a biconditional.
       (Contributed by NM, 8-Sep-2006.) $)
    3anbi2i $p |- ( ( ch /\ ph /\ th ) <-> ( ch /\ ps /\ th ) ) $=
      ( biid 3anbi123i ) CCABDDCFEDFG $.

    $( Inference adding two conjuncts to each side of a biconditional.
       (Contributed by NM, 8-Sep-2006.) $)
    3anbi3i $p |- ( ( ch /\ th /\ ph ) <-> ( ch /\ th /\ ps ) ) $=
      ( biid 3anbi123i ) CCDDABCFDFEG $.
  $}

  ${
    3imp.1 $e |- ( ph -> ( ps -> ( ch -> th ) ) ) $.
    $( Importation inference.  (Contributed by NM, 8-Apr-1994.) $)
    3imp $p |- ( ( ph /\ ps /\ ch ) -> th ) $=
      ( w3a wa df-3an imp31 sylbi ) ABCFABGCGDABCHABCDEIJ $.
  $}

  ${
    3impa.1 $e |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $.
    $( Importation from double to triple conjunction.  (Contributed by NM,
       20-Aug-1995.) $)
    3impa $p |- ( ( ph /\ ps /\ ch ) -> th ) $=
      ( exp31 3imp ) ABCDABCDEFG $.
  $}

  ${
    3impb.1 $e |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $.
    $( Importation from double to triple conjunction.  (Contributed by NM,
       20-Aug-1995.) $)
    3impb $p |- ( ( ph /\ ps /\ ch ) -> th ) $=
      ( exp32 3imp ) ABCDABCDEFG $.
  $}

  ${
    3impia.1 $e |- ( ( ph /\ ps ) -> ( ch -> th ) ) $.
    $( Importation to triple conjunction.  (Contributed by NM, 13-Jun-2006.) $)
    3impia $p |- ( ( ph /\ ps /\ ch ) -> th ) $=
      ( wi ex 3imp ) ABCDABCDFEGH $.
  $}

  ${
    3impib.1 $e |- ( ph -> ( ( ps /\ ch ) -> th ) ) $.
    $( Importation to triple conjunction.  (Contributed by NM, 13-Jun-2006.) $)
    3impib $p |- ( ( ph /\ ps /\ ch ) -> th ) $=
      ( expd 3imp ) ABCDABCDEFG $.
  $}

  ${
    3exp.1 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( Exportation inference.  (Contributed by NM, 30-May-1994.) $)
    3exp $p |- ( ph -> ( ps -> ( ch -> th ) ) ) $=
      ( w3a pm3.2an3 syl8 ) ABCABCFDABCGEH $.

    $( Exportation from triple to double conjunction.  (Contributed by NM,
       20-Aug-1995.) $)
    3expa $p |- ( ( ( ph /\ ps ) /\ ch ) -> th ) $=
      ( 3exp imp31 ) ABCDABCDEFG $.

    $( Exportation from triple to double conjunction.  (Contributed by NM,
       20-Aug-1995.) $)
    3expb $p |- ( ( ph /\ ( ps /\ ch ) ) -> th ) $=
      ( 3exp imp32 ) ABCDABCDEFG $.

    $( Exportation from triple conjunction.  (Contributed by NM,
       19-May-2007.) $)
    3expia $p |- ( ( ph /\ ps ) -> ( ch -> th ) ) $=
      ( wi 3exp imp ) ABCDFABCDEGH $.

    $( Exportation from triple conjunction.  (Contributed by NM,
       19-May-2007.) $)
    3expib $p |- ( ph -> ( ( ps /\ ch ) -> th ) ) $=
      ( 3exp impd ) ABCDABCDEFG $.

    $( Commutation in antecedent.  Swap 1st and 3rd.  (Contributed by NM,
       28-Jan-1996.)  (Proof shortened by Andrew Salmon, 13-May-2011.) $)
    3com12 $p |- ( ( ps /\ ph /\ ch ) -> th ) $=
      ( w3a 3ancoma sylbi ) BACFABCFDBACGEH $.

    $( Commutation in antecedent.  Swap 1st and 3rd.  (Contributed by NM,
       28-Jan-1996.) $)
    3com13 $p |- ( ( ch /\ ps /\ ph ) -> th ) $=
      ( w3a 3anrev sylbi ) CBAFABCFDCBAGEH $.

    $( Commutation in antecedent.  Swap 2nd and 3rd.  (Contributed by NM,
       28-Jan-1996.) $)
    3com23 $p |- ( ( ph /\ ch /\ ps ) -> th ) $=
      ( 3exp com23 3imp ) ACBDABCDABCDEFGH $.

    $( Commutation in antecedent.  Rotate left.  (Contributed by NM,
       28-Jan-1996.) $)
    3coml $p |- ( ( ps /\ ch /\ ph ) -> th ) $=
      ( 3com23 3com13 ) ACBDABCDEFG $.

    $( Commutation in antecedent.  Rotate right.  (Contributed by NM,
       28-Jan-1996.) $)
    3comr $p |- ( ( ch /\ ph /\ ps ) -> th ) $=
      ( 3coml ) BCADABCDEFF $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       16-Feb-2008.) $)
    3adant3r1 $p |- ( ( ph /\ ( ta /\ ps /\ ch ) ) -> th ) $=
      ( 3expb 3adantr1 ) ABCDEABCDFGH $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       17-Feb-2008.) $)
    3adant3r2 $p |- ( ( ph /\ ( ps /\ ta /\ ch ) ) -> th ) $=
      ( 3expb 3adantr2 ) ABCDEABCDFGH $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       18-Feb-2008.) $)
    3adant3r3 $p |- ( ( ph /\ ( ps /\ ch /\ ta ) ) -> th ) $=
      ( 3expb 3adantr3 ) ABCDEABCDFGH $.
  $}

  ${
    3an1rs.1 $e |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ta ) $.
    $( Swap conjuncts.  (Contributed by NM, 16-Dec-2007.) $)
    3an1rs $p |- ( ( ( ph /\ ps /\ th ) /\ ch ) -> ta ) $=
      ( w3a wi ex 3exp com34 3imp imp ) ABDGCEABDCEHABCDEABCDEHABCGDEFIJKLM $.
  $}

  ${
    3imp1.1 $e |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $.
    $( Importation to left triple conjunction.  (Contributed by NM,
       24-Feb-2005.) $)
    3imp1 $p |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ta ) $=
      ( w3a wi 3imp imp ) ABCGDEABCDEHFIJ $.

    $( Importation deduction for triple conjunction.  (Contributed by NM,
       26-Oct-2006.) $)
    3impd $p |- ( ph -> ( ( ps /\ ch /\ th ) -> ta ) ) $=
      ( w3a wi com4l 3imp com12 ) BCDGAEBCDAEHABCDEFIJK $.

    $( Importation to right triple conjunction.  (Contributed by NM,
       26-Oct-2006.) $)
    3imp2 $p |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ta ) $=
      ( w3a 3impd imp ) ABCDGEABCDEFHI $.
  $}

  ${
    3exp1.1 $e |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ta ) $.
    $( Exportation from left triple conjunction.  (Contributed by NM,
       24-Feb-2005.) $)
    3exp1 $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wi w3a ex 3exp ) ABCDEGABCHDEFIJ $.
  $}

  ${
    3expd.1 $e |- ( ph -> ( ( ps /\ ch /\ th ) -> ta ) ) $.
    $( Exportation deduction for triple conjunction.  (Contributed by NM,
       26-Oct-2006.) $)
    3expd $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( wi w3a com12 3exp com4r ) BCDAEBCDAEGABCDHEFIJK $.
  $}

  ${
    3exp2.1 $e |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ta ) $.
    $( Exportation from right triple conjunction.  (Contributed by NM,
       26-Oct-2006.) $)
    3exp2 $p |- ( ph -> ( ps -> ( ch -> ( th -> ta ) ) ) ) $=
      ( w3a ex 3expd ) ABCDEABCDGEFHI $.
  $}

  ${
    exp5o.1 $e |- ( ( ph /\ ps /\ ch ) -> ( ( th /\ ta ) -> et ) ) $.
    $( A triple exportation inference.  (Contributed by Jeff Hankins,
       8-Jul-2009.) $)
    exp5o $p |- ( ph -> ( ps -> ( ch -> ( th -> ( ta -> et ) ) ) ) ) $=
      ( wi w3a expd 3exp ) ABCDEFHHABCIDEFGJK $.
  $}

  ${
    exp516.1 $e |- ( ( ( ph /\ ( ps /\ ch /\ th ) ) /\ ta ) -> et ) $.
    $( A triple exportation inference.  (Contributed by Jeff Hankins,
       8-Jul-2009.) $)
    exp516 $p |- ( ph -> ( ps -> ( ch -> ( th -> ( ta -> et ) ) ) ) ) $=
      ( wi w3a exp31 3expd ) ABCDEFHABCDIEFGJK $.
  $}

  ${
    exp520.1 $e |- ( ( ( ph /\ ps /\ ch ) /\ ( th /\ ta ) ) -> et ) $.
    $( A triple exportation inference.  (Contributed by Jeff Hankins,
       8-Jul-2009.) $)
    exp520 $p |- ( ph -> ( ps -> ( ch -> ( th -> ( ta -> et ) ) ) ) ) $=
      ( w3a wa ex exp5o ) ABCDEFABCHDEIFGJK $.
  $}

  ${
    3anassrs.1 $e |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ta ) $.
    $( Associative law for conjunction applied to antecedent (eliminates
       syllogism).  (Contributed by Mario Carneiro, 4-Jan-2017.) $)
    3anassrs $p |- ( ( ( ( ph /\ ps ) /\ ch ) /\ th ) -> ta ) $=
      ( 3exp2 imp41 ) ABCDEABCDEFGH $.
  $}

  $( Associative law for four conjunctions with a triple conjunction.
     (Contributed by Alexander van der Vekens, 24-Jun-2018.) $)
  3an4anass $p |- ( ( ( ph /\ ps /\ ch ) /\ th )
                    <-> ( ( ph /\ ps ) /\ ( ch /\ th ) ) ) $=
    ( w3a wa df-3an anbi1i anass bitri ) ABCEZDFABFZCFZDFLCDFFKMDABCGHLCDIJ $.

  ${
    3adant1l.1 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       8-Jan-2006.) $)
    3adant1l $p |- ( ( ( ta /\ ph ) /\ ps /\ ch ) -> th ) $=
      ( wa 3expb adantll 3impb ) EAGBCDABCGDEABCDFHIJ $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       8-Jan-2006.) $)
    3adant1r $p |- ( ( ( ph /\ ta ) /\ ps /\ ch ) -> th ) $=
      ( wa 3expb adantlr 3impb ) AEGBCDABCGDEABCDFHIJ $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       8-Jan-2006.) $)
    3adant2l $p |- ( ( ph /\ ( ta /\ ps ) /\ ch ) -> th ) $=
      ( wa 3com12 3adant1l ) EBGACDBACDEABCDFHIH $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       8-Jan-2006.) $)
    3adant2r $p |- ( ( ph /\ ( ps /\ ta ) /\ ch ) -> th ) $=
      ( wa 3com12 3adant1r ) BEGACDBACDEABCDFHIH $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       8-Jan-2006.) $)
    3adant3l $p |- ( ( ph /\ ps /\ ( ta /\ ch ) ) -> th ) $=
      ( wa 3com13 3adant1l ) ECGBADCBADEABCDFHIH $.

    $( Deduction adding a conjunct to antecedent.  (Contributed by NM,
       8-Jan-2006.) $)
    3adant3r $p |- ( ( ph /\ ps /\ ( ch /\ ta ) ) -> th ) $=
      ( wa 3com13 3adant1r ) CEGBADCBADEABCDFHIH $.
  $}

  ${
    sylXanc.1 $e |- ( ph -> ps ) $.
    sylXanc.2 $e |- ( ph -> ch ) $.
    sylXanc.3 $e |- ( ph -> th ) $.
    ${
      syl12anc.4 $e |- ( ( ps /\ ( ch /\ th ) ) -> ta ) $.
      $( Syllogism combined with contraction.  (Contributed by Jeff Hankins,
         1-Aug-2009.) $)
      syl12anc $p |- ( ph -> ta ) $=
        ( wa jca32 syl ) ABCDJJEABCDFGHKIL $.
    $}

    ${
      syl21anc.4 $e |- ( ( ( ps /\ ch ) /\ th ) -> ta ) $.
      $( Syllogism combined with contraction.  (Contributed by Jeff Hankins,
         1-Aug-2009.) $)
      syl21anc $p |- ( ph -> ta ) $=
        ( wa jca31 syl ) ABCJDJEABCDFGHKIL $.
    $}

    ${
      syl111anc.4 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl3anc $p |- ( ph -> ta ) $=
        ( w3a 3jca syl ) ABCDJEABCDFGHKIL $.
    $}

    sylXanc.4 $e |- ( ph -> ta ) $.
    ${
      syl22anc.5 $e |- ( ( ( ps /\ ch ) /\ ( th /\ ta ) ) -> et ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl22anc $p |- ( ph -> et ) $=
        ( wa jca syl12anc ) ABCLDEFABCGHMIJKN $.
    $}

    ${
      syl13anc.5 $e |- ( ( ps /\ ( ch /\ th /\ ta ) ) -> et ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl13anc $p |- ( ph -> et ) $=
        ( w3a 3jca syl2anc ) ABCDELFGACDEHIJMKN $.
    $}

    ${
      syl31anc.5 $e |- ( ( ( ps /\ ch /\ th ) /\ ta ) -> et ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl31anc $p |- ( ph -> et ) $=
        ( w3a 3jca syl2anc ) ABCDLEFABCDGHIMJKN $.
    $}

    ${
      syl112anc.5 $e |- ( ( ps /\ ch /\ ( th /\ ta ) ) -> et ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl112anc $p |- ( ph -> et ) $=
        ( wa jca syl3anc ) ABCDELFGHADEIJMKN $.
    $}

    ${
      syl121anc.5 $e |- ( ( ps /\ ( ch /\ th ) /\ ta ) -> et ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl121anc $p |- ( ph -> et ) $=
        ( wa jca syl3anc ) ABCDLEFGACDHIMJKN $.
    $}

    ${
      syl211anc.5 $e |- ( ( ( ps /\ ch ) /\ th /\ ta ) -> et ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl211anc $p |- ( ph -> et ) $=
        ( wa jca syl3anc ) ABCLDEFABCGHMIJKN $.
    $}

    sylXanc.5 $e |- ( ph -> et ) $.
    ${
      syl23anc.6 $e |- ( ( ( ps /\ ch ) /\ ( th /\ ta /\ et ) ) -> ze ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl23anc $p |- ( ph -> ze ) $=
        ( wa jca syl13anc ) ABCNDEFGABCHIOJKLMP $.
    $}

    ${
      syl32anc.6 $e |- ( ( ( ps /\ ch /\ th ) /\ ( ta /\ et ) ) -> ze ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl32anc $p |- ( ph -> ze ) $=
        ( wa jca syl31anc ) ABCDEFNGHIJAEFKLOMP $.
    $}

    ${
      syl122anc.6 $e |- ( ( ps /\ ( ch /\ th ) /\ ( ta /\ et ) ) -> ze ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl122anc $p |- ( ph -> ze ) $=
        ( wa jca syl121anc ) ABCDEFNGHIJAEFKLOMP $.
    $}

    ${
      syl212anc.6 $e |- ( ( ( ps /\ ch ) /\ th /\ ( ta /\ et ) ) -> ze ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl212anc $p |- ( ph -> ze ) $=
        ( wa jca syl211anc ) ABCDEFNGHIJAEFKLOMP $.
    $}

    ${
      syl221anc.6 $e |- ( ( ( ps /\ ch ) /\ ( th /\ ta ) /\ et ) -> ze ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl221anc $p |- ( ph -> ze ) $=
        ( wa jca syl211anc ) ABCDENFGHIADEJKOLMP $.
    $}

    ${
      syl113anc.6 $e |- ( ( ps /\ ch /\ ( th /\ ta /\ et ) ) -> ze ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl113anc $p |- ( ph -> ze ) $=
        ( w3a 3jca syl3anc ) ABCDEFNGHIADEFJKLOMP $.
    $}

    ${
      syl131anc.6 $e |- ( ( ps /\ ( ch /\ th /\ ta ) /\ et ) -> ze ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl131anc $p |- ( ph -> ze ) $=
        ( w3a 3jca syl3anc ) ABCDENFGHACDEIJKOLMP $.
    $}

    ${
      syl311anc.6 $e |- ( ( ( ps /\ ch /\ th ) /\ ta /\ et ) -> ze ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl311anc $p |- ( ph -> ze ) $=
        ( w3a 3jca syl3anc ) ABCDNEFGABCDHIJOKLMP $.
    $}

    sylXanc.6 $e |- ( ph -> ze ) $.
    ${
      syl33anc.7 $e |- ( ( ( ps /\ ch /\ th ) /\ ( ta /\ et /\ ze ) )
           -> si ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl33anc $p |- ( ph -> si ) $=
        ( w3a 3jca syl13anc ) ABCDPEFGHABCDIJKQLMNOR $.
    $}

    ${
      syl222anc.7 $e |- ( ( ( ps /\ ch ) /\ ( th /\ ta ) /\ ( et /\ ze ) )
           -> si ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl222anc $p |- ( ph -> si ) $=
        ( wa jca syl221anc ) ABCDEFGPHIJKLAFGMNQOR $.
    $}

    ${
      syl123anc.7 $e |- ( ( ps /\ ( ch /\ th ) /\ ( ta /\ et /\ ze ) )
           -> si ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl123anc $p |- ( ph -> si ) $=
        ( wa jca syl113anc ) ABCDPEFGHIACDJKQLMNOR $.
    $}

    ${
      syl132anc.7 $e |- ( ( ps /\ ( ch /\ th /\ ta ) /\ ( et /\ ze ) )
           -> si ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Jul-2012.) $)
      syl132anc $p |- ( ph -> si ) $=
        ( wa jca syl131anc ) ABCDEFGPHIJKLAFGMNQOR $.
    $}

    ${
      syl213anc.7 $e |- ( ( ( ps /\ ch ) /\ th /\ ( ta /\ et /\ ze ) )
           -> si ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl213anc $p |- ( ph -> si ) $=
        ( wa jca syl113anc ) ABCPDEFGHABCIJQKLMNOR $.
    $}

    ${
      syl231anc.7 $e |- ( ( ( ps /\ ch ) /\ ( th /\ ta /\ et ) /\ ze )
           -> si ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl231anc $p |- ( ph -> si ) $=
        ( wa jca syl131anc ) ABCPDEFGHABCIJQKLMNOR $.
    $}

    ${
      syl312anc.7 $e |- ( ( ( ps /\ ch /\ th ) /\ ta /\ ( et /\ ze ) )
           -> si ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Jul-2012.) $)
      syl312anc $p |- ( ph -> si ) $=
        ( wa jca syl311anc ) ABCDEFGPHIJKLAFGMNQOR $.
    $}

    ${
      syl321anc.7 $e |- ( ( ( ps /\ ch /\ th ) /\ ( ta /\ et ) /\ ze )
           -> si ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Jul-2012.) $)
      syl321anc $p |- ( ph -> si ) $=
        ( wa jca syl311anc ) ABCDEFPGHIJKAEFLMQNOR $.
    $}

    sylXanc.7 $e |- ( ph -> si ) $.
    ${
      syl133anc.8 $e |- ( ( ps /\ ( ch /\ th /\ ta ) /\ ( et /\ ze /\ si ) )
           -> rh ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl133anc $p |- ( ph -> rh ) $=
        ( w3a 3jca syl131anc ) ABCDEFGHRIJKLMAFGHNOPSQT $.
    $}

    ${
      syl313anc.8 $e |- ( ( ( ps /\ ch /\ th ) /\ ta /\ ( et /\ ze /\ si ) )
           -> rh ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl313anc $p |- ( ph -> rh ) $=
        ( w3a 3jca syl311anc ) ABCDEFGHRIJKLMAFGHNOPSQT $.
    $}

    ${
      syl331anc.8 $e |- ( ( ( ps /\ ch /\ th ) /\ ( ta /\ et /\ ze ) /\ si )
           -> rh ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl331anc $p |- ( ph -> rh ) $=
        ( w3a 3jca syl311anc ) ABCDEFGRHIJKLAEFGMNOSPQT $.
    $}

    ${
      syl223anc.8 $e |- ( ( ( ps /\ ch ) /\ ( th /\ ta ) /\ ( et /\ ze /\ si )
          ) -> rh ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl223anc $p |- ( ph -> rh ) $=
        ( wa jca syl213anc ) ABCDERFGHIJKADELMSNOPQT $.
    $}

    ${
      syl232anc.8 $e |- ( ( ( ps /\ ch ) /\ ( th /\ ta /\ et ) /\ ( ze /\ si )
          ) -> rh ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl232anc $p |- ( ph -> rh ) $=
        ( wa jca syl231anc ) ABCDEFGHRIJKLMNAGHOPSQT $.
    $}

    ${
      syl322anc.8 $e |- ( ( ( ps /\ ch /\ th ) /\ ( ta /\ et ) /\ ( ze /\ si )
          ) -> rh ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl322anc $p |- ( ph -> rh ) $=
        ( wa jca syl321anc ) ABCDEFGHRIJKLMNAGHOPSQT $.
    $}

    sylXanc.8 $e |- ( ph -> rh ) $.
    ${
      syl233anc.9 $e |- ( ( ( ps /\ ch ) /\ ( th /\ ta /\ et ) /\ ( ze /\ si /\
          rh ) ) -> mu ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl233anc $p |- ( ph -> mu ) $=
        ( wa jca syl133anc ) ABCTDEFGHIJABCKLUAMNOPQRSUB $.
    $}

    ${
      syl323anc.9 $e |- ( ( ( ps /\ ch /\ th ) /\ ( ta /\ et ) /\ ( ze /\ si /\
          rh ) ) -> mu ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl323anc $p |- ( ph -> mu ) $=
        ( wa jca syl313anc ) ABCDEFTGHIJKLMAEFNOUAPQRSUB $.
    $}

    ${
      syl332anc.9 $e |- ( ( ( ps /\ ch /\ th ) /\ ( ta /\ et /\ ze ) /\ ( si /\
          rh ) ) -> mu ) $.
      $( Syllogism combined with contraction.  (Contributed by NM,
         11-Mar-2012.) $)
      syl332anc $p |- ( ph -> mu ) $=
        ( wa jca syl331anc ) ABCDEFGHITJKLMNOPAHIQRUASUB $.
    $}

    sylXanc.9 $e |- ( ph -> mu ) $.
    ${
      syl333anc.10 $e |- ( ( ( ps /\ ch /\ th ) /\ ( ta /\ et /\ ze )
          /\ ( si /\ rh /\ mu ) ) -> la ) $.
      $( A syllogism inference combined with contraction.  (Contributed by NM,
         10-Mar-2012.) $)
      syl333anc $p |- ( ph -> la ) $=
        ( w3a 3jca syl331anc ) ABCDEFGHIJUBKLMNOPQAHIJRSTUCUAUD $.
    $}
  $}

  ${
    syl3an1.1 $e |- ( ph -> ps ) $.
    syl3an1.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an1 $p |- ( ( ph /\ ch /\ th ) -> ta ) $=
      ( w3a 3anim1i syl ) ACDHBCDHEABCDFIGJ $.
  $}

  ${
    syl3an2.1 $e |- ( ph -> ch ) $.
    syl3an2.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an2 $p |- ( ( ps /\ ph /\ th ) -> ta ) $=
      ( wi 3exp syl5 3imp ) BADEACBDEHFBCDEGIJK $.
  $}

  ${
    syl3an3.1 $e |- ( ph -> th ) $.
    syl3an3.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an3 $p |- ( ( ps /\ ch /\ ph ) -> ta ) $=
      ( 3exp syl7 3imp ) BCAEADBCEFBCDEGHIJ $.
  $}

  ${
    syl3an1b.1 $e |- ( ph <-> ps ) $.
    syl3an1b.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an1b $p |- ( ( ph /\ ch /\ th ) -> ta ) $=
      ( biimpi syl3an1 ) ABCDEABFHGI $.
  $}

  ${
    syl3an2b.1 $e |- ( ph <-> ch ) $.
    syl3an2b.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an2b $p |- ( ( ps /\ ph /\ th ) -> ta ) $=
      ( biimpi syl3an2 ) ABCDEACFHGI $.
  $}

  ${
    syl3an3b.1 $e |- ( ph <-> th ) $.
    syl3an3b.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an3b $p |- ( ( ps /\ ch /\ ph ) -> ta ) $=
      ( biimpi syl3an3 ) ABCDEADFHGI $.
  $}

  ${
    syl3an1br.1 $e |- ( ps <-> ph ) $.
    syl3an1br.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an1br $p |- ( ( ph /\ ch /\ th ) -> ta ) $=
      ( biimpri syl3an1 ) ABCDEBAFHGI $.
  $}

  ${
    syl3an2br.1 $e |- ( ch <-> ph ) $.
    syl3an2br.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an2br $p |- ( ( ps /\ ph /\ th ) -> ta ) $=
      ( biimpri syl3an2 ) ABCDECAFHGI $.
  $}

  ${
    syl3an3br.1 $e |- ( th <-> ph ) $.
    syl3an3br.2 $e |- ( ( ps /\ ch /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 22-Aug-1995.) $)
    syl3an3br $p |- ( ( ps /\ ch /\ ph ) -> ta ) $=
      ( biimpri syl3an3 ) ABCDEDAFHGI $.
  $}

  ${
    syl3an.1 $e |- ( ph -> ps ) $.
    syl3an.2 $e |- ( ch -> th ) $.
    syl3an.3 $e |- ( ta -> et ) $.
    syl3an.4 $e |- ( ( ps /\ th /\ et ) -> ze ) $.
    $( A triple syllogism inference.  (Contributed by NM, 13-May-2004.) $)
    syl3an $p |- ( ( ph /\ ch /\ ta ) -> ze ) $=
      ( w3a 3anim123i syl ) ACELBDFLGABCDEFHIJMKN $.
  $}

  ${
    syl3anb.1 $e |- ( ph <-> ps ) $.
    syl3anb.2 $e |- ( ch <-> th ) $.
    syl3anb.3 $e |- ( ta <-> et ) $.
    syl3anb.4 $e |- ( ( ps /\ th /\ et ) -> ze ) $.
    $( A triple syllogism inference.  (Contributed by NM, 15-Oct-2005.) $)
    syl3anb $p |- ( ( ph /\ ch /\ ta ) -> ze ) $=
      ( w3a 3anbi123i sylbi ) ACELBDFLGABCDEFHIJMKN $.
  $}

  ${
    syl3anbr.1 $e |- ( ps <-> ph ) $.
    syl3anbr.2 $e |- ( th <-> ch ) $.
    syl3anbr.3 $e |- ( et <-> ta ) $.
    syl3anbr.4 $e |- ( ( ps /\ th /\ et ) -> ze ) $.
    $( A triple syllogism inference.  (Contributed by NM, 29-Dec-2011.) $)
    syl3anbr $p |- ( ( ph /\ ch /\ ta ) -> ze ) $=
      ( bicomi syl3anb ) ABCDEFGBAHLDCILFEJLKM $.
  $}

  ${
    syld3an3.1 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    syld3an3.2 $e |- ( ( ph /\ ps /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 20-May-2007.) $)
    syld3an3 $p |- ( ( ph /\ ps /\ ch ) -> ta ) $=
      ( w3a simp1 simp2 syl3anc ) ABCHABDEABCIABCJFGK $.
  $}

  ${
    syld3an1.1 $e |- ( ( ch /\ ps /\ th ) -> ph ) $.
    syld3an1.2 $e |- ( ( ph /\ ps /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 7-Jul-2008.) $)
    syld3an1 $p |- ( ( ch /\ ps /\ th ) -> ta ) $=
      ( 3com13 syld3an3 ) DBCEDBCAECBDAFHABDEGHIH $.
  $}

  ${
    syld3an2.1 $e |- ( ( ph /\ ch /\ th ) -> ps ) $.
    syld3an2.2 $e |- ( ( ph /\ ps /\ th ) -> ta ) $.
    $( A syllogism inference.  (Contributed by NM, 20-May-2007.) $)
    syld3an2 $p |- ( ( ph /\ ch /\ th ) -> ta ) $=
      ( 3com23 syld3an3 ) ADCEADCBEACDBFHABDEGHIH $.
  $}

  ${
    syl3anl1.1 $e |- ( ph -> ps ) $.
    syl3anl1.2 $e |- ( ( ( ps /\ ch /\ th ) /\ ta ) -> et ) $.
    $( A syllogism inference.  (Contributed by NM, 24-Feb-2005.) $)
    syl3anl1 $p |- ( ( ( ph /\ ch /\ th ) /\ ta ) -> et ) $=
      ( w3a 3anim1i sylan ) ACDIBCDIEFABCDGJHK $.
  $}

  ${
    syl3anl2.1 $e |- ( ph -> ch ) $.
    syl3anl2.2 $e |- ( ( ( ps /\ ch /\ th ) /\ ta ) -> et ) $.
    $( A syllogism inference.  (Contributed by NM, 24-Feb-2005.) $)
    syl3anl2 $p |- ( ( ( ps /\ ph /\ th ) /\ ta ) -> et ) $=
      ( w3a wi ex syl3an2 imp ) BADIEFABCDEFJGBCDIEFHKLM $.
  $}

  ${
    syl3anl3.1 $e |- ( ph -> th ) $.
    syl3anl3.2 $e |- ( ( ( ps /\ ch /\ th ) /\ ta ) -> et ) $.
    $( A syllogism inference.  (Contributed by NM, 24-Feb-2005.) $)
    syl3anl3 $p |- ( ( ( ps /\ ch /\ ph ) /\ ta ) -> et ) $=
      ( w3a 3anim3i sylan ) BCAIBCDIEFADBCGJHK $.
  $}

  ${
    syl3anl.1 $e |- ( ph -> ps ) $.
    syl3anl.2 $e |- ( ch -> th ) $.
    syl3anl.3 $e |- ( ta -> et ) $.
    syl3anl.4 $e |- ( ( ( ps /\ th /\ et ) /\ ze ) -> si ) $.
    $( A triple syllogism inference.  (Contributed by NM, 24-Dec-2006.) $)
    syl3anl $p |- ( ( ( ph /\ ch /\ ta ) /\ ze ) -> si ) $=
      ( w3a 3anim123i sylan ) ACEMBDFMGHABCDEFIJKNLO $.
  $}

  ${
    syl3anr1.1 $e |- ( ph -> ps ) $.
    syl3anr1.2 $e |- ( ( ch /\ ( ps /\ th /\ ta ) ) -> et ) $.
    $( A syllogism inference.  (Contributed by NM, 31-Jul-2007.) $)
    syl3anr1 $p |- ( ( ch /\ ( ph /\ th /\ ta ) ) -> et ) $=
      ( w3a 3anim1i sylan2 ) ADEICBDEIFABDEGJHK $.
  $}

  ${
    syl3anr2.1 $e |- ( ph -> th ) $.
    syl3anr2.2 $e |- ( ( ch /\ ( ps /\ th /\ ta ) ) -> et ) $.
    $( A syllogism inference.  (Contributed by NM, 1-Aug-2007.) $)
    syl3anr2 $p |- ( ( ch /\ ( ps /\ ph /\ ta ) ) -> et ) $=
      ( w3a ancoms syl3anl2 ) BAEICFABDECFGCBDEIFHJKJ $.
  $}

  ${
    syl3anr3.1 $e |- ( ph -> ta ) $.
    syl3anr3.2 $e |- ( ( ch /\ ( ps /\ th /\ ta ) ) -> et ) $.
    $( A syllogism inference.  (Contributed by NM, 23-Aug-2007.) $)
    syl3anr3 $p |- ( ( ch /\ ( ps /\ th /\ ph ) ) -> et ) $=
      ( w3a 3anim3i sylan2 ) BDAICBDEIFAEBDGJHK $.
  $}

  ${
    3impdi.1 $e |- ( ( ( ph /\ ps ) /\ ( ph /\ ch ) ) -> th ) $.
    $( Importation inference (undistribute conjunction).  (Contributed by NM,
       14-Aug-1995.) $)
    3impdi $p |- ( ( ph /\ ps /\ ch ) -> th ) $=
      ( anandis 3impb ) ABCDABCDEFG $.
  $}

  ${
    3impdir.1 $e |- ( ( ( ph /\ ps ) /\ ( ch /\ ps ) ) -> th ) $.
    $( Importation inference (undistribute conjunction).  (Contributed by NM,
       20-Aug-1995.) $)
    3impdir $p |- ( ( ph /\ ch /\ ps ) -> th ) $=
      ( anandirs 3impa ) ACBDACBDEFG $.
  $}

  ${
    3anidm12.1 $e |- ( ( ph /\ ph /\ ps ) -> ch ) $.
    $( Inference from idempotent law for conjunction.  (Contributed by NM,
       7-Mar-2008.) $)
    3anidm12 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( 3expib anabsi5 ) ABCAABCDEF $.
  $}

  ${
    3anidm13.1 $e |- ( ( ph /\ ps /\ ph ) -> ch ) $.
    $( Inference from idempotent law for conjunction.  (Contributed by NM,
       7-Mar-2008.) $)
    3anidm13 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( 3com23 3anidm12 ) ABCABACDEF $.
  $}

  ${
    3anidm23.1 $e |- ( ( ph /\ ps /\ ps ) -> ch ) $.
    $( Inference from idempotent law for conjunction.  (Contributed by NM,
       1-Feb-2007.) $)
    3anidm23 $p |- ( ( ph /\ ps ) -> ch ) $=
      ( 3expa anabss3 ) ABCABBCDEF $.
  $}

  ${
    3ori.1 $e |- ( ph \/ ps \/ ch ) $.
    $( Infer implication from triple disjunction.  (Contributed by NM,
       26-Sep-2006.) $)
    3ori $p |- ( ( -. ph /\ -. ps ) -> ch ) $=
      ( wn wa wo ioran w3o df-3or mpbi ori sylbir ) AEBEFABGZECABHNCABCINCGDABC
      JKLM $.
  $}

  $( Disjunction of 3 antecedents.  (Contributed by NM, 8-Apr-1994.) $)
  3jao $p |- ( ( ( ph -> ps ) /\ ( ch -> ps ) /\ ( th -> ps ) ) ->
              ( ( ph \/ ch \/ th ) -> ps ) ) $=
    ( w3o wo wi w3a df-3or jao syl6 3imp syl5bi ) ACDEACFZDFZABGZCBGZDBGZHBACDI
    PQROBGZPQNBGRSGABCJNBDJKLM $.

  $( Disjunction of 3 antecedents.  (Contributed by NM, 13-Sep-2011.) $)
  3jaob $p |- ( ( ( ph \/ ch \/ th ) -> ps ) <->
              ( ( ph -> ps ) /\ ( ch -> ps ) /\ ( th -> ps ) ) ) $=
    ( w3o wi w3a 3mix1 imim1i 3mix2 3mix3 3jca 3jao impbii ) ACDEZBFZABFZCBFZDB
    FZGPQRSAOBACDHICOBCADJIDOBDACKILABCDMN $.

  ${
    3jaoi.1 $e |- ( ph -> ps ) $.
    3jaoi.2 $e |- ( ch -> ps ) $.
    3jaoi.3 $e |- ( th -> ps ) $.
    $( Disjunction of 3 antecedents (inference).  (Contributed by NM,
       12-Sep-1995.) $)
    3jaoi $p |- ( ( ph \/ ch \/ th ) -> ps ) $=
      ( wi w3a w3o 3pm3.2i 3jao ax-mp ) ABHZCBHZDBHZIACDJBHNOPEFGKABCDLM $.
  $}

  ${
    3jaod.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3jaod.2 $e |- ( ph -> ( th -> ch ) ) $.
    3jaod.3 $e |- ( ph -> ( ta -> ch ) ) $.
    $( Disjunction of 3 antecedents (deduction).  (Contributed by NM,
       14-Oct-2005.) $)
    3jaod $p |- ( ph -> ( ( ps \/ th \/ ta ) -> ch ) ) $=
      ( wi w3o 3jao syl3anc ) ABCIDCIECIBDEJCIFGHBCDEKL $.
  $}

  ${
    3jaoian.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    3jaoian.2 $e |- ( ( th /\ ps ) -> ch ) $.
    3jaoian.3 $e |- ( ( ta /\ ps ) -> ch ) $.
    $( Disjunction of 3 antecedents (inference).  (Contributed by NM,
       14-Oct-2005.) $)
    3jaoian $p |- ( ( ( ph \/ th \/ ta ) /\ ps ) -> ch ) $=
      ( w3o wi ex 3jaoi imp ) ADEIBCABCJDEABCFKDBCGKEBCHKLM $.
  $}

  ${
    3jaodan.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    3jaodan.2 $e |- ( ( ph /\ th ) -> ch ) $.
    3jaodan.3 $e |- ( ( ph /\ ta ) -> ch ) $.
    $( Disjunction of 3 antecedents (deduction).  (Contributed by NM,
       14-Oct-2005.) $)
    3jaodan $p |- ( ( ph /\ ( ps \/ th \/ ta ) ) -> ch ) $=
      ( w3o ex 3jaod imp ) ABDEICABCDEABCFJADCGJAECHJKL $.
  $}

  ${
    mpjao3dan.1 $e |- ( ( ph /\ ps ) -> ch ) $.
    mpjao3dan.2 $e |- ( ( ph /\ th ) -> ch ) $.
    mpjao3dan.3 $e |- ( ( ph /\ ta ) -> ch ) $.
    mpjao3dan.4 $e |- ( ph -> ( ps \/ th \/ ta ) ) $.
    $( Eliminate a 3-way disjunction in a deduction.  (Contributed by Thierry
       Arnoux, 13-Apr-2018.) $)
    mpjao3dan $p |- ( ph -> ch ) $=
      ( wo jaodan w3o df-3or sylib mpjaodan ) ABDJZCEABCDFGKHABDELPEJIBDEMNO $.
  $}

  ${
    3jaao.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3jaao.2 $e |- ( th -> ( ta -> ch ) ) $.
    3jaao.3 $e |- ( et -> ( ze -> ch ) ) $.
    $( Inference conjoining and disjoining the antecedents of three
       implications.  (Contributed by Jeff Hankins, 15-Aug-2009.)  (Proof
       shortened by Andrew Salmon, 13-May-2011.) $)
    3jaao $p |- ( ( ph /\ th /\ et ) -> ( ( ps \/ ta \/ ze ) -> ch ) ) $=
      ( w3a wi 3ad2ant1 3ad2ant2 3ad2ant3 3jaod ) ADFKBCEGADBCLFHMDAECLFINFAGCL
      DJOP $.
  $}

  ${
    syl3an9b.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    syl3an9b.2 $e |- ( th -> ( ch <-> ta ) ) $.
    syl3an9b.3 $e |- ( et -> ( ta <-> ze ) ) $.
    $( Nested syllogism inference conjoining 3 dissimilar antecedents.
       (Contributed by NM, 1-May-1995.) $)
    syl3an9b $p |- ( ( ph /\ th /\ et ) -> ( ps <-> ze ) ) $=
      ( wb wa sylan9bb 3impa ) ADFBGKADLBEFGABCDEHIMJMN $.
  $}

  ${
    bi3d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    bi3d.2 $e |- ( ph -> ( th <-> ta ) ) $.
    bi3d.3 $e |- ( ph -> ( et <-> ze ) ) $.
    $( Deduction joining 3 equivalences to form equivalence of disjunctions.
       (Contributed by NM, 20-Apr-1994.) $)
    3orbi123d $p |- ( ph -> ( ( ps \/ th \/ et ) <-> ( ch \/ ta \/ ze ) ) ) $=
      ( wo w3o orbi12d df-3or 3bitr4g ) ABDKZFKCEKZGKBDFLCEGLAPQFGABCDEHIMJMBDF
      NCEGNO $.

    $( Deduction joining 3 equivalences to form equivalence of conjunctions.
       (Contributed by NM, 22-Apr-1994.) $)
    3anbi123d $p |- ( ph -> ( ( ps /\ th /\ et ) <-> ( ch /\ ta /\ ze ) ) ) $=
      ( wa w3a anbi12d df-3an 3bitr4g ) ABDKZFKCEKZGKBDFLCEGLAPQFGABCDEHIMJMBDF
      NCEGNO $.
  $}

  ${
    3anbi12d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    3anbi12d.2 $e |- ( ph -> ( th <-> ta ) ) $.
    $( Deduction conjoining and adding a conjunct to equivalences.
       (Contributed by NM, 8-Sep-2006.) $)
    3anbi12d $p |- ( ph -> ( ( ps /\ th /\ et ) <-> ( ch /\ ta /\ et ) ) ) $=
      ( biidd 3anbi123d ) ABCDEFFGHAFIJ $.

    $( Deduction conjoining and adding a conjunct to equivalences.
       (Contributed by NM, 8-Sep-2006.) $)
    3anbi13d $p |- ( ph -> ( ( ps /\ et /\ th ) <-> ( ch /\ et /\ ta ) ) ) $=
      ( biidd 3anbi123d ) ABCFFDEGAFIHJ $.

    $( Deduction conjoining and adding a conjunct to equivalences.
       (Contributed by NM, 8-Sep-2006.) $)
    3anbi23d $p |- ( ph -> ( ( et /\ ps /\ th ) <-> ( et /\ ch /\ ta ) ) ) $=
      ( biidd 3anbi123d ) AFFBCDEAFIGHJ $.
  $}

  ${
    3anbi1d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Deduction adding conjuncts to an equivalence.  (Contributed by NM,
       8-Sep-2006.) $)
    3anbi1d $p |- ( ph -> ( ( ps /\ th /\ ta ) <-> ( ch /\ th /\ ta ) ) ) $=
      ( biidd 3anbi12d ) ABCDDEFADGH $.

    $( Deduction adding conjuncts to an equivalence.  (Contributed by NM,
       8-Sep-2006.) $)
    3anbi2d $p |- ( ph -> ( ( th /\ ps /\ ta ) <-> ( th /\ ch /\ ta ) ) ) $=
      ( biidd 3anbi12d ) ADDBCEADGFH $.

    $( Deduction adding conjuncts to an equivalence.  (Contributed by NM,
       8-Sep-2006.) $)
    3anbi3d $p |- ( ph -> ( ( th /\ ta /\ ps ) <-> ( th /\ ta /\ ch ) ) ) $=
      ( biidd 3anbi13d ) ADDBCEADGFH $.
  $}

  ${
    3anim123d.1 $e |- ( ph -> ( ps -> ch ) ) $.
    3anim123d.2 $e |- ( ph -> ( th -> ta ) ) $.
    3anim123d.3 $e |- ( ph -> ( et -> ze ) ) $.
    $( Deduction joining 3 implications to form implication of conjunctions.
       (Contributed by NM, 24-Feb-2005.) $)
    3anim123d $p |- ( ph -> ( ( ps /\ th /\ et ) -> ( ch /\ ta /\ ze ) ) ) $=
      ( wa w3a anim12d df-3an 3imtr4g ) ABDKZFKCEKZGKBDFLCEGLAPQFGABCDEHIMJMBDF
      NCEGNO $.

    $( Deduction joining 3 implications to form implication of disjunctions.
       (Contributed by NM, 4-Apr-1997.) $)
    3orim123d $p |- ( ph -> ( ( ps \/ th \/ et ) -> ( ch \/ ta \/ ze ) ) ) $=
      ( wo w3o orim12d df-3or 3imtr4g ) ABDKZFKCEKZGKBDFLCEGLAPQFGABCDEHIMJMBDF
      NCEGNO $.
  $}

  $( Rearrangement of 6 conjuncts.  (Contributed by NM, 13-Mar-1995.) $)
  an6 $p |- ( ( ( ph /\ ps /\ ch ) /\ ( th /\ ta /\ et ) ) <->
              ( ( ph /\ th ) /\ ( ps /\ ta ) /\ ( ch /\ et ) ) ) $=
    ( wa w3a an4 anbi1i bitri df-3an anbi12i 3bitr4i ) ABGZCGZDEGZFGZGZADGZBEGZ
    GZCFGZGZABCHZDEFHZGTUAUCHSOQGZUCGUDOCQFIUGUBUCABDEIJKUEPUFRABCLDEFLMTUAUCLN
    $.

  $( Analog of ~ an4 for triple conjunction.  (Contributed by Scott Fenton,
     16-Mar-2011.)  (Proof shortened by Andrew Salmon, 25-May-2011.) $)
  3an6 $p |- ( ( ( ph /\ ps ) /\ ( ch /\ th ) /\ ( ta /\ et ) ) <->
                ( ( ph /\ ch /\ ta ) /\ ( ps /\ th /\ et ) ) ) $=
    ( w3a wa an6 bicomi ) ACEGBDFGHABHCDHEFHGACEBDFIJ $.

  $( Analog of ~ or4 for triple conjunction.  (Contributed by Scott Fenton,
     16-Mar-2011.) $)
  3or6 $p |- ( ( ( ph \/ ps ) \/ ( ch \/ th ) \/ ( ta \/ et ) ) <->
                ( ( ph \/ ch \/ ta ) \/ ( ps \/ th \/ et ) ) ) $=
    ( wo w3o or4 orbi1i bitr2i df-3or orbi12i 3bitr4i ) ABGZCDGZGZEFGZGZACGZEGZ
    BDGZFGZGZOPRHACEHZBDFHZGUDTUBGZRGSTEUBFIUGQRACBDIJKOPRLUEUAUFUCACELBDFLMN
    $.

  ${
    mp3an1.1 $e |- ph $.
    mp3an1.2 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       21-Nov-1994.) $)
    mp3an1 $p |- ( ( ps /\ ch ) -> th ) $=
      ( wa 3expb mpan ) ABCGDEABCDFHI $.
  $}

  ${
    mp3an2.1 $e |- ps $.
    mp3an2.2 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       21-Nov-1994.) $)
    mp3an2 $p |- ( ( ph /\ ch ) -> th ) $=
      ( 3expa mpanl2 ) ABCDEABCDFGH $.
  $}

  ${
    mp3an3.1 $e |- ch $.
    mp3an3.2 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       21-Nov-1994.) $)
    mp3an3 $p |- ( ( ph /\ ps ) -> th ) $=
      ( wa 3expia mpi ) ABGCDEABCDFHI $.
  $}

  ${
    mp3an12.1 $e |- ph $.
    mp3an12.2 $e |- ps $.
    mp3an12.3 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       13-Jul-2005.) $)
    mp3an12 $p |- ( ch -> th ) $=
      ( mp3an1 mpan ) BCDFABCDEGHI $.
  $}

  ${
    mp3an13.1 $e |- ph $.
    mp3an13.2 $e |- ch $.
    mp3an13.3 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       14-Jul-2005.) $)
    mp3an13 $p |- ( ps -> th ) $=
      ( mp3an3 mpan ) ABDEABCDFGHI $.
  $}

  ${
    mp3an23.1 $e |- ps $.
    mp3an23.2 $e |- ch $.
    mp3an23.3 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       14-Jul-2005.) $)
    mp3an23 $p |- ( ph -> th ) $=
      ( mp3an3 mpan2 ) ABDEABCDFGHI $.
  $}

  ${
    mp3an1i.1 $e |- ps $.
    mp3an1i.2 $e |- ( ph -> ( ( ps /\ ch /\ th ) -> ta ) ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 5-Jul-2005.) $)
    mp3an1i $p |- ( ph -> ( ( ch /\ th ) -> ta ) ) $=
      ( wa wi w3a com12 mp3an1 ) CDHAEBCDAEIFABCDJEGKLK $.
  $}

  ${
    mp3anl1.1 $e |- ph $.
    mp3anl1.2 $e |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ta ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       24-Feb-2005.) $)
    mp3anl1 $p |- ( ( ( ps /\ ch ) /\ th ) -> ta ) $=
      ( wa wi w3a ex mp3an1 imp ) BCHDEABCDEIFABCJDEGKLM $.
  $}

  ${
    mp3anl2.1 $e |- ps $.
    mp3anl2.2 $e |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ta ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       24-Feb-2005.) $)
    mp3anl2 $p |- ( ( ( ph /\ ch ) /\ th ) -> ta ) $=
      ( wa wi w3a ex mp3an2 imp ) ACHDEABCDEIFABCJDEGKLM $.
  $}

  ${
    mp3anl3.1 $e |- ch $.
    mp3anl3.2 $e |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ta ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       24-Feb-2005.) $)
    mp3anl3 $p |- ( ( ( ph /\ ps ) /\ th ) -> ta ) $=
      ( wa wi w3a ex mp3an3 imp ) ABHDEABCDEIFABCJDEGKLM $.
  $}

  ${
    mp3anr1.1 $e |- ps $.
    mp3anr1.2 $e |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ta ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 4-Nov-2006.) $)
    mp3anr1 $p |- ( ( ph /\ ( ch /\ th ) ) -> ta ) $=
      ( wa w3a ancoms mp3anl1 ) CDHAEBCDAEFABCDIEGJKJ $.
  $}

  ${
    mp3anr2.1 $e |- ch $.
    mp3anr2.2 $e |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ta ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       24-Nov-2006.) $)
    mp3anr2 $p |- ( ( ph /\ ( ps /\ th ) ) -> ta ) $=
      ( wa w3a ancoms mp3anl2 ) BDHAEBCDAEFABCDIEGJKJ $.
  $}

  ${
    mp3anr3.1 $e |- th $.
    mp3anr3.2 $e |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ta ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       19-Oct-2007.) $)
    mp3anr3 $p |- ( ( ph /\ ( ps /\ ch ) ) -> ta ) $=
      ( wa w3a ancoms mp3anl3 ) BCHAEBCDAEFABCDIEGJKJ $.
  $}

  ${
    mp3an.1 $e |- ph $.
    mp3an.2 $e |- ps $.
    mp3an.3 $e |- ch $.
    mp3an.4 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM,
       14-May-1999.) $)
    mp3an $p |- th $=
      ( mp3an1 mp2an ) BCDFGABCDEHIJ $.
  $}

  ${
    mpd3an3.2 $e |- ( ( ph /\ ps ) -> ch ) $.
    mpd3an3.3 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 8-Nov-2007.) $)
    mpd3an3 $p |- ( ( ph /\ ps ) -> th ) $=
      ( wa 3expa mpdan ) ABGCDEABCDFHI $.
  $}

  ${
    mpd3an23.1 $e |- ( ph -> ps ) $.
    mpd3an23.2 $e |- ( ph -> ch ) $.
    mpd3an23.3 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( An inference based on modus ponens.  (Contributed by NM, 4-Dec-2006.) $)
    mpd3an23 $p |- ( ph -> th ) $=
      ( id syl3anc ) AABCDAHEFGI $.
  $}

  ${
    mp3and.1 $e |- ( ph -> ps ) $.
    mp3and.2 $e |- ( ph -> ch ) $.
    mp3and.3 $e |- ( ph -> th ) $.
    mp3and.4 $e |- ( ph -> ( ( ps /\ ch /\ th ) -> ta ) ) $.
    $( A deduction based on modus ponens.  (Contributed by Mario Carneiro,
       24-Dec-2016.) $)
    mp3and $p |- ( ph -> ta ) $=
      ( w3a 3jca mpd ) ABCDJEABCDFGHKIL $.
  $}

  ${
    biimp3a.1 $e |- ( ( ph /\ ps ) -> ( ch <-> th ) ) $.
    $( Infer implication from a logical equivalence.  Similar to ~ biimpa .
       (Contributed by NM, 4-Sep-2005.) $)
    biimp3a $p |- ( ( ph /\ ps /\ ch ) -> th ) $=
      ( wa biimpa 3impa ) ABCDABFCDEGH $.

    $( Infer implication from a logical equivalence.  Similar to ~ biimpar .
       (Contributed by NM, 2-Jan-2009.) $)
    biimp3ar $p |- ( ( ph /\ ps /\ th ) -> ch ) $=
      ( exbiri 3imp ) ABDCABCDEFG $.
  $}

  ${
    3anandis.1 $e |- ( ( ( ph /\ ps ) /\ ( ph /\ ch ) /\ ( ph /\ th ) )
                      -> ta ) $.
    $( Inference that undistributes a triple conjunction in the antecedent.
       (Contributed by NM, 18-Apr-2007.) $)
    3anandis $p |- ( ( ph /\ ( ps /\ ch /\ th ) ) -> ta ) $=
      ( w3a wa simpl simpr1 simpr2 simpr3 syl222anc ) ABCDGZHABACADEANIZABCDJOA
      BCDKOABCDLFM $.
  $}

  ${
    3anandirs.1 $e |- ( ( ( ph /\ th ) /\ ( ps /\ th ) /\ ( ch /\ th ) )
                      -> ta ) $.
    $( Inference that undistributes a triple conjunction in the antecedent.
       (Contributed by NM, 25-Jul-2006.) $)
    3anandirs $p |- ( ( ( ph /\ ps /\ ch ) /\ th ) -> ta ) $=
      ( w3a wa simpl1 simpr simpl2 simpl3 syl222anc ) ABCGZDHADBDCDEABCDINDJZAB
      CDKOABCDLOFM $.
  $}

  ${
    ecase23d.1 $e |- ( ph -> -. ch ) $.
    ecase23d.2 $e |- ( ph -> -. th ) $.
    ecase23d.3 $e |- ( ph -> ( ps \/ ch \/ th ) ) $.
    $( Deduction for elimination by cases.  (Contributed by NM,
       22-Apr-1994.) $)
    ecase23d $p |- ( ph -> ps ) $=
      ( wo wn ioran sylanbrc w3o 3orass sylib ord mt3d ) ABCDHZACIDIQIEFCDJKABQ
      ABCDLBQHGBCDMNOP $.
  $}

  ${
    3ecase.1 $e |- ( -. ph -> th ) $.
    3ecase.2 $e |- ( -. ps -> th ) $.
    3ecase.3 $e |- ( -. ch -> th ) $.
    3ecase.4 $e |- ( ( ph /\ ps /\ ch ) -> th ) $.
    $( Inference for elimination by cases.  (Contributed by NM,
       13-Jul-2005.) $)
    3ecase $p |- th $=
      ( wi 3exp wn a1d pm2.61i pm2.61nii ) BCDABCDIZIABCDHJAKZOBPDCELLMFGN $.
  $}

  ${
    3biorfd.1 $e |- ( ph -> -. th ) $.
    $( A disjunction is equivalent to a threefold disjunction with single
       falsehood, analogous to ~ biorf .  (Contributed by Alexander van der
       Vekens, 8-Sep-2017.) $)
    3bior1fd $p |- ( ph -> ( ( ch \/ ps ) <-> ( th \/ ch \/ ps ) ) ) $=
      ( wo w3o wn wb biorf syl 3orass syl6bbr ) ACBFZDNFZDCBGADHNOIEDNJKDCBLM
      $.

    $( A disjunction is equivalent to a threefold disjunction with single
       falsehood of a conjunction.  (Contributed by Alexander van der Vekens,
       8-Sep-2017.) $)
    3bior1fand $p |- ( ph -> ( ( ch \/ ps )
                       <-> ( ( th /\ ta ) \/ ch \/ ps ) ) ) $=
      ( wa intnanrd 3bior1fd ) ABCDEGADEFHI $.

    3biorfd.2 $e |- ( ph -> -. ch ) $.
    $( A wff is equivalent to its threefold disjunction with double falsehood,
       analogous to ~ biorf .  (Contributed by Alexander van der Vekens,
       8-Sep-2017.) $)
    3bior2fd $p |- ( ph -> ( ps <-> ( th \/ ch \/ ps ) ) ) $=
      ( wo w3o wn wb biorf syl 3bior1fd bitrd ) ABCBGZDCBHACIBOJFCBKLABCDEMN $.
  $}

  ${
    3biantd.1 $e |- ( ph -> th ) $.
    $( A conjunction is equivalent to a threefold conjunction with single
       truth, analogous to ~ biantrud .  (Contributed by Alexander van der
       Vekens, 26-Sep-2017.) $)
    3biant1d $p |- ( ph -> ( ( ch /\ ps ) <-> ( th /\ ch /\ ps ) ) ) $=
      ( wa w3a biantrurd 3anass syl6bbr ) ACBFZDKFDCBGADKEHDCBIJ $.
  $}

  ${
    intn3and.1 $e |- ( ph -> -. ps ) $.
    $( Introduction of a triple conjunct inside a contradiction.  (Contributed
       by FL, 27-Dec-2007.)  (Proof shortened by Andrew Salmon,
       26-Jun-2011.) $)
    intn3an1d $p |- ( ph -> -. ( ps /\ ch /\ th ) ) $=
      ( w3a simp1 nsyl ) ABBCDFEBCDGH $.

    $( Introduction of a triple conjunct inside a contradiction.  (Contributed
       by FL, 27-Dec-2007.)  (Proof shortened by Andrew Salmon,
       26-Jun-2011.) $)
    intn3an2d $p |- ( ph -> -. ( ch /\ ps /\ th ) ) $=
      ( w3a simp2 nsyl ) ABCBDFECBDGH $.

    $( Introduction of a triple conjunct inside a contradiction.  (Contributed
       by FL, 27-Dec-2007.)  (Proof shortened by Andrew Salmon,
       26-Jun-2011.) $)
    intn3an3d $p |- ( ph -> -. ( ch /\ th /\ ps ) ) $=
      ( w3a simp3 nsyl ) ABCDBFECDBGH $.
  $}

  $( Distribution of conjunction over conjunction.  (Contributed by Thierry
     Arnoux, 8-Apr-2019.) $)
  an3andi $p |- ( ( ph /\ ( ps /\ ch /\ th ) ) <->
               ( ( ph /\ ps ) /\ ( ph /\ ch ) /\ ( ph /\ th ) ) ) $=
    ( w3a wa df-3an anbi2i anandi anbi1i 3bitri bitr4i ) ABCDEZFZABFZACFZFZADFZ
    FZOPRENABCFZDFZFATFZRFSMUAABCDGHATDIUBQRABCIJKOPRGL $.

  $( Rearrange 9-fold conjunction.  (Contributed by Thierry Arnoux,
     14-Apr-2019.) $)
  an33rean $p |- (
        ( ( ph /\ ps /\ ch ) /\ ( th /\ ta /\ et ) /\ ( ze /\ si /\ rh ) ) <->
    ( ( ph /\ ta /\ rh ) /\ ( ( ps /\ th ) /\ ( et /\ si ) /\ ( ch /\ ze ) ) )
    ) $=
    ( w3a wa 3anass 3anan12 3an6 an4 anbi2i 3bitr4i anbi1i df-3an 3bitri 3anrev
    bitri 3anbi123i 3ancomb ) ABCJZDEFJZGHIJZJABCKZKZEDFKZKZIHGKZKZJAEIJZUHUJUL
    JZKUNBDKZFHKCGKJZKUEUIUFUKUGUMABCLDEFMUGIHGJUMGHIUAIHGLUBUCAUHEUJIULNUOUQUN
    UOUHDHKZFGKZJZUPCHKZUSJZUQUHUJULKZKUHURUSKZKUOUTVCVDUHDFHGOPUHUJULLUHURUSLQ
    UHURKZUSKUPVAKZUSKUTVBVEVFUSBCDHORUHURUSSUPVAUSSQBCFJZDHGJZKBFCJZVHKVBUQVGV
    IVHBCFUDRBDCHFGNBDFHCGNQTPT $.

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Logical 'nand' (Sheffer stroke)
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
$)

  $( Declare connective for alternative denial ('nand'). $)
  $c -/\ $. $( Overlined 'wedge' (read:  'nand') $)

  $( Extend wff definition to include alternative denial ('nand'). $)
  wnan $a wff ( ph -/\ ps ) $.

  $( Define incompatibility, or alternative denial ('not-and' or 'nand').  This
     is also called the Sheffer stroke, represented by a vertical bar, but we
     use a different symbol to avoid ambiguity with other uses of the vertical
     bar.  In the second edition of Principia Mathematica (1927), Russell and
     Whitehead used the Sheffer stroke and suggested it as a replacement for
     the "or" and "not" operations of the first edition.  However, in practice,
     "or" and "not" are more widely used.  After we define the constant true
     ` T. ` ( ~ df-tru ) and the constant false ` F. ` ( ~ df-fal ), we will be
     able to prove these truth table values: ` ( ( T. -/\ T. ) <-> F. ) `
     ( ~ trunantru ), ` ( ( T. -/\ F. ) <-> T. ) ` ( ~ trunanfal ),
     ` ( ( F. -/\ T. ) <-> T. ) ` ( ~ falnantru ), and
     ` ( ( F. -/\ F. ) <-> T. ) ` ( ~ falnanfal ).  Contrast with ` /\ `
     ( ~ df-an ), ` \/ ` ( ~ df-or ), ` -> ` ( ~ wi ), and ` \/_ `
     ( ~ df-xor ) .  (Contributed by Jeff Hoffman, 19-Nov-2007.) $)
  df-nan $a |- ( ( ph -/\ ps ) <-> -. ( ph /\ ps ) ) $.

  $( Write 'and' in terms of 'nand'.  (Contributed by Mario Carneiro,
     9-May-2015.) $)
  nanan $p |- ( ( ph /\ ps ) <-> -. ( ph -/\ ps ) ) $=
    ( wnan wa df-nan con2bii ) ABCABDABEF $.

  $( The 'nand' operator commutes.  (Contributed by Mario Carneiro,
     9-May-2015.)  (Proof shortened by Wolf Lammen, 7-Mar-2020.) $)
  nancom $p |- ( ( ph -/\ ps ) <-> ( ps -/\ ph ) ) $=
    ( wnan wa wn df-nan ancom xchbinx bitr4i ) ABCZBADZEBACJABDKABFABGHBAFI $.

  $( Obsolete proof of ~ nancom as of 7-Mar-2020.  (Contributed by Mario
     Carneiro, 9-May-2015.)  (Proof modification is discouraged.)
     (New usage is discouraged.) $)
  nancomOLD $p |- ( ( ph -/\ ps ) <-> ( ps -/\ ph ) ) $=
    ( wa wn wnan ancom notbii df-nan 3bitr4i ) ABCZDBACZDABEBAEJKABFGABHBAHI $.

  $( Lemma for handling nested 'nand's.  (Contributed by Jeff Hoffman,
     19-Nov-2007.)  (Proof shortened by Wolf Lammen, 7-Mar-2020.) $)
  nannan $p |- ( ( ph -/\ ( ch -/\ ps ) ) <-> ( ph -> ( ch /\ ps ) ) ) $=
    ( wnan wn wi wa imnan nanan imbi2i df-nan 3bitr4ri ) ACBDZEZFAMGEACBGZFAMDA
    MHONACBIJAMKL $.

  $( Obsolete proof of ~ nannan as of 7-Mar-2020.  (Contributed by Jeff
     Hoffman, 19-Nov-2007.)  (Proof modification is discouraged.)
     (New usage is discouraged.) $)
  nannanOLD $p |- ( ( ph -/\ ( ch -/\ ps ) ) <-> ( ph -> ( ch /\ ps ) ) ) $=
    ( wnan wa wn wi df-nan anbi2i xchbinx iman bitr4i ) ACBDZDZACBEZFZEZFAOGNAM
    EQAMHMPACBHIJAOKL $.

  $( Show equivalence between implication and the Nicod version.  To derive
     ~ nic-dfim , apply ~ nanbi .  (Contributed by Jeff Hoffman,
     19-Nov-2007.) $)
  nanim $p |- ( ( ph -> ps ) <-> ( ph -/\ ( ps -/\ ps ) ) ) $=
    ( wnan wa wi nannan anidmdbi bitr2i ) ABBCCABBDEABEABBFABGH $.

  $( Show equivalence between negation and the Nicod version.  To derive
     ~ nic-dfneg , apply ~ nanbi .  (Contributed by Jeff Hoffman,
     19-Nov-2007.) $)
  nannot $p |- ( -. ps <-> ( ps -/\ ps ) ) $=
    ( wnan wn wa df-nan anidm xchbinx bicomi ) AABZACIAADAAAEAFGH $.

  $( Show equivalence between the biconditional and the Nicod version.
     (Contributed by Jeff Hoffman, 19-Nov-2007.)  (Proof shortened by Wolf
     Lammen, 9-Mar-2020.) $)
  nanbi $p |- ( ( ph <-> ps ) <->
          ( ( ph -/\ ps ) -/\ ( ( ph -/\ ph ) -/\ ( ps -/\ ps ) ) ) ) $=
    ( wb wnan wa wi wn df-or dfbi3 df-nan nannot anbi12i bicomi imbi12i 3bitr4i
    wo nannan bitr4i ) ABCZABDZAADZBBDZEZFZTUAUBDDABEZAGZBGZEZPUEGZUHFSUDUEUHHA
    BITUIUCUHABJUHUCUFUAUGUBAKBKLMNOTUBUAQR $.

  $( Obsolete proof of ~ nanbi as of 8-Mar-2020.  (Contributed by Jeff Hoffman,
     19-Nov-2007.)  (Proof modification is discouraged.)
     (New usage is discouraged.) $)
  nanbiOLD $p |- ( ( ph <-> ps ) <->
          ( ( ph -/\ ps ) -/\ ( ( ph -/\ ph ) -/\ ( ps -/\ ps ) ) ) ) $=
    ( wa wn wo wb pm4.57 df-nan nannot anbi12i xchbinxr xchbinx dfbi3 3bitr4ri
    wnan ) ABCZDZADZBDZCZDZCZDPTEABOZAAOZBBOZOZOZABFPTGUGUCUFCUBUCUFHUCQUFUAABH
    UFUDUECTUDUEHRUDSUEAIBIJKJLABMN $.

  ${
    $( Introduce a right anti-conjunct to both sides of a logical equivalence.
       (Contributed by SF, 2-Jan-2018.) $)
    nanbi1 $p |- ( ( ph <-> ps ) -> ( ( ph -/\ ch ) <-> ( ps -/\ ch ) ) ) $=
      ( wb wa wn wnan anbi1 notbid df-nan 3bitr4g ) ABDZACEZFBCEZFACGBCGLMNABCH
      IACJBCJK $.

    $( Introduce a left anti-conjunct to both sides of a logical equivalence.
       (Contributed by SF, 2-Jan-2018.) $)
    nanbi2 $p |- ( ( ph <-> ps ) -> ( ( ch -/\ ph ) <-> ( ch -/\ ps ) ) ) $=
      ( wb wnan nanbi1 nancom 3bitr4g ) ABDACEBCECAECBEABCFCAGCBGH $.

    $( Join two logical equivalences with anti-conjunction.  (Contributed by
       SF, 2-Jan-2018.) $)
    nanbi12 $p |- ( ( ( ph <-> ps ) /\ ( ch <-> th ) ) ->
       ( ( ph -/\ ch ) <-> ( ps -/\ th ) ) ) $=
      ( wb wnan nanbi1 nanbi2 sylan9bb ) ABEACFBCFCDEBDFABCGCDBHI $.
  $}

  ${
    nanbii.1 $e |- ( ph <-> ps ) $.
    $( Introduce a right anti-conjunct to both sides of a logical equivalence.
       (Contributed by SF, 2-Jan-2018.) $)
    nanbi1i $p |- ( ( ph -/\ ch ) <-> ( ps -/\ ch ) ) $=
      ( wb wnan nanbi1 ax-mp ) ABEACFBCFEDABCGH $.

    $( Introduce a left anti-conjunct to both sides of a logical equivalence.
       (Contributed by SF, 2-Jan-2018.) $)
    nanbi2i $p |- ( ( ch -/\ ph ) <-> ( ch -/\ ps ) ) $=
      ( wb wnan nanbi2 ax-mp ) ABECAFCBFEDABCGH $.

    nanbi12i.2 $e |- ( ch <-> th ) $.
    $( Join two logical equivalences with anti-conjunction.  (Contributed by
       SF, 2-Jan-2018.) $)
    nanbi12i $p |- ( ( ph -/\ ch ) <-> ( ps -/\ th ) ) $=
      ( wb wnan nanbi12 mp2an ) ABGCDGACHBDHGEFABCDIJ $.
  $}

  ${
    nanbid.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    $( Introduce a right anti-conjunct to both sides of a logical equivalence.
       (Contributed by SF, 2-Jan-2018.) $)
    nanbi1d $p |- ( ph -> ( ( ps -/\ th ) <-> ( ch -/\ th ) ) ) $=
      ( wb wnan nanbi1 syl ) ABCFBDGCDGFEBCDHI $.

    $( Introduce a left anti-conjunct to both sides of a logical equivalence.
       (Contributed by SF, 2-Jan-2018.) $)
    nanbi2d $p |- ( ph -> ( ( th -/\ ps ) <-> ( th -/\ ch ) ) ) $=
      ( wb wnan nanbi2 syl ) ABCFDBGDCGFEBCDHI $.

    nanbi12d.2 $e |- ( ph -> ( th <-> ta ) ) $.
    $( Join two logical equivalences with anti-conjunction.  (Contributed by
       Scott Fenton, 2-Jan-2018.) $)
    nanbi12d $p |- ( ph -> ( ( ps -/\ th ) <-> ( ch -/\ ta ) ) ) $=
      ( wb wnan nanbi12 syl2anc ) ABCHDEHBDICEIHFGBCDEJK $.
  $}

$(
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        Logical 'xor'
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
$)

  $( Declare connective for exclusive disjunction ('xor'). $)
  $c \/_ $. $( Underlined 'vee' (read:  'xor') $)

  $( Extend wff definition to include exclusive disjunction ('xor'). $)
  wxo $a wff ( ph \/_ ps ) $.

  $( Define exclusive disjunction (logical 'xor').  Return true if either the
     left or right, but not both, are true.  After we define the constant true
     ` T. ` ( ~ df-tru ) and the constant false ` F. ` ( ~ df-fal ), we will be
     able to prove these truth table values: ` ( ( T. \/_ T. ) <-> F. ) `
     ( ~ truxortru ), ` ( ( T. \/_ F. ) <-> T. ) ` ( ~ truxorfal ),
     ` ( ( F. \/_ T. ) <-> T. ) ` ( ~ falxortru ), and
     ` ( ( F. \/_ F. ) <-> F. ) ` ( ~ falxorfal ).  Contrast with ` /\ `
     ( ~ df-an ), ` \/ ` ( ~ df-or ), ` -> ` ( ~ wi ), and ` -/\ `
     ( ~ df-nan ) .  (Contributed by FL, 22-Nov-2010.) $)
  df-xor $a |- ( ( ph \/_ ps ) <-> -. ( ph <-> ps ) ) $.

  $( Two ways to write XNOR. (Contributed by Mario Carneiro, 4-Sep-2016.) $)
  xnor $p |- ( ( ph <-> ps ) <-> -. ( ph \/_ ps ) ) $=
    ( wxo wb df-xor con2bii ) ABCABDABEF $.

  $( ` \/_ ` is commutative.  (Contributed by Mario Carneiro, 4-Sep-2016.) $)
  xorcom $p |- ( ( ph \/_ ps ) <-> ( ps \/_ ph ) ) $=
    ( wb wn wxo bicom notbii df-xor 3bitr4i ) ABCZDBACZDABEBAEJKABFGABHBAHI $.

  $( ` \/_ ` is associative.  (Contributed by FL, 22-Nov-2010.)  (Proof
     shortened by Andrew Salmon, 8-Jun-2011.) $)
  xorass $p |- ( ( ( ph \/_ ps ) \/_ ch ) <-> ( ph \/_ ( ps \/_ ch ) ) ) $=
    ( wxo wb wn biass notbii nbbn pm5.18 con2bii 3bitr4i df-xor bibi1i bibi2i )
    ABDZCEZFABCDZEZFPCDARDQSABEZFZCEZABCEZFZEZQSTCEZFAUCEZFUBUEUFUGABCGHTCIUGUE
    AUCJKLPUACABMNRUDABCMOLHPCMARML $.

  $( This tautology shows that xor is really exclusive.  (Contributed by FL,
     22-Nov-2010.) $)
  excxor $p |- ( ( ph \/_ ps ) <->
       ( ( ph /\ -. ps ) \/ ( -. ph /\ ps ) ) ) $=
    ( wxo wb wn wa wo df-xor xor ancom orbi2i 3bitri ) ABCABDEABEFZBAEZFZGMNBFZ
    GABHABIOPMBNJKL $.

  $( Two ways to express "exclusive or."  (Contributed by Mario Carneiro,
     4-Sep-2016.) $)
  xor2 $p |- ( ( ph \/_ ps ) <->
       ( ( ph \/ ps ) /\ -. ( ph /\ ps ) ) ) $=
    ( wxo wb wn wo wa df-xor nbi2 bitri ) ABCABDEABFABGEGABHABIJ $.

  $( XOR implies OR. (Contributed by BJ, 19-Apr-2019.) $)
  xoror $p |- ( ( ph \/_ ps ) -> ( ph \/ ps ) ) $=
    ( wxo wo wa wn xor2 simplbi ) ABCABDABEFABGH $.

  $( XOR implies NAND. (Contributed by BJ, 19-Apr-2019.) $)
  xornan $p |- ( ( ph \/_ ps ) -> -. ( ph /\ ps ) ) $=
    ( wxo wo wa wn xor2 simprbi ) ABCABDABEFABGH $.

  $( XOR implies NAND (written with the ` -/\ ` connector).  (Contributed by
     BJ, 19-Apr-2019.) $)
  xornan2 $p |- ( ( ph \/_ ps ) -> ( ph -/\ ps ) ) $=
    ( wxo wa wn wnan xornan df-nan sylibr ) ABCABDEABFABGABHI $.

  $( ` \/_ ` is negated under negation of one argument.  (Contributed by Mario
     Carneiro, 4-Sep-2016.) $)
  xorneg1 $p |- ( ( -. ph \/_ ps ) <-> -. ( ph \/_ ps ) ) $=
    ( wn wxo wb df-xor nbbn con2bii xnor 3bitr2i ) ACZBDKBEZCABEZABDCKBFLMABGHA
    BIJ $.

  $( ` \/_ ` is negated under negation of one argument.  (Contributed by Mario
     Carneiro, 4-Sep-2016.) $)
  xorneg2 $p |- ( ( ph \/_ -. ps ) <-> -. ( ph \/_ ps ) ) $=
    ( wn wxo xorneg1 xorcom notbii 3bitr4i ) BCZADBADZCAIDABDZCBAEAIFKJABFGH $.

  $( ` \/_ ` is unchanged under negation of both arguments.  (Contributed by
     Mario Carneiro, 4-Sep-2016.) $)
  xorneg $p |- ( ( -. ph \/_ -. ps ) <-> ( ph \/_ ps ) ) $=
    ( wn wxo xorneg1 xorneg2 con2bii bitr4i ) ACBCZDAIDZCABDZAIEJKABFGH $.

  ${
    xorbi12.1 $e |- ( ph <-> ps ) $.
    xorbi12.2 $e |- ( ch <-> th ) $.
    $( Equality property for XOR. (Contributed by Mario Carneiro,
       4-Sep-2016.) $)
    xorbi12i $p |- ( ( ph \/_ ch ) <-> ( ps \/_ th ) ) $=
      ( wb wn wxo bibi12i notbii df-xor 3bitr4i ) ACGZHBDGZHACIBDINOABCDEFJKACL
      BDLM $.
  $}

  ${
    xor12d.1 $e |- ( ph -> ( ps <-> ch ) ) $.
    xor12d.2 $e |- ( ph -> ( th <-> ta ) ) $.
    $( Equality property for XOR. (Contributed by Mario Carneiro,
       4-Sep-2016.) $)
    xorbi12d $p |- ( ph -> ( ( ps \/_ th ) <-> ( ch \/_ ta ) ) ) $=
      ( wb wn wxo bibi12d notbid df-xor 3bitr4g ) ABDHZICEHZIBDJCEJAOPABCDEFGKL
      BDMCEMN $.
  $}

  $( Conjunction distributes over exclusive-or.  In intuitionistic logic this
     assertion is also true, even though ~ xordi does not necessarily hold, in
     part because the usual definition of xor is subtly different in
     intuitionistic logic.  (Contributed by David A. Wheeler, 7-Oct-2018.) $)
  anxordi $p |- ( ( ph /\ ( ps \/_ ch ) ) <->
      ( ( ph /\ ps ) \/_ ( ph /\ ch ) ) ) $=
    ( wb wn wa wxo xordi df-xor anbi2i 3bitr4i ) ABCDEZFABFZACFZDEABCGZFMNGABCH
    OLABCIJMNIK $.

  $( Exclusive-or variant of the law of the excluded middle ( ~ exmid ).  This
     statement is ancient, going back to at least Stoic logic.  This statement
     does not necessarily hold in intuitionistic logic.  (Contributed by David
     A. Wheeler, 23-Feb-2019.) $)
  xorexmid $p |- ( ph \/_ -. ph ) $=
    ( wn wxo wb pm5.19 df-xor mpbir ) AABZCAHDBAEAHFG $.
