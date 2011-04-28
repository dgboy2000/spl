# open FILE, "filename.txt" or die $!;
# my @objv1 = [];
# my @objv2 = [];
# while (<FILE>) {
#   print $_;
# }
# close(FILE);

sub get_objectives_array {
  my $filename = $_[0];
  
  open FILE, $filename or die $!;
  my @objv = ();
  while (<FILE>) {
    if (/^(\S+)\s+\S+/) {
      push (@objv, $1);
    }
  }
  close(FILE);
  
  \@objv;
}

my $num_args = scalar @ARGV;
$num_args == 2 or die "Need to have 2 arguments to compare_time_files.pl, supplied $num_args";

my $objv0_ref = get_objectives_array($ARGV[0]);
my $objv1_ref = get_objectives_array($ARGV[1]);

my @objv0 = @$objv0_ref;
my @objv1 = @$objv1_ref;

my $num0 = scalar (@objv0);
my $num1 = scalar (@objv1);
$num0 == $num1 or die "FAIL: $num0 iterations in $ARGV[0] NOT EQUAL to $num1 in $ARGV[1]";
for (my $ind = 0; $ind <= $#objv0; $ind++) {
  $objv0[$ind] == $objv1[$ind] or die "FAIL: iteration $ind objectives differ in $ARGV[0] and $ARGV[1]: $objv0[$ind] vs. $objv1[$ind]";
}

print "PASS: equal objectives in $ARGV[0] and $ARGV[1]\n";