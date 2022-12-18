from skbio import Protein
from skbio.alignment import global_pairwise_align_protein

def align_and_count(peptide1, peptide2):
    aln, _, _ = global_pairwise_align_protein(
        Protein(peptide1),
        Protein(peptide2)
    )

    seq1, seq2 = aln
    print(seq1, seq2)
    return seq1.match_frequency(seq2, relative=True)

print(align_and_count("DCEKRAR", "QAHCHRAR"))

