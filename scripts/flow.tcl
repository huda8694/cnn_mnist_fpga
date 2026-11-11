# -----------------------------------------------------------------------------
# flow.tcl - complete Vivado flow for the Nexys A7-100T MNIST CNN project
#
#   vivado -mode batch -source scripts/flow.tcl -tclargs build     (default)
#       create project -> synthesis -> implementation -> bitstream  (output/mnist_top.bit)
#   vivado -mode batch -source scripts/flow.tcl -tclargs sim
#       run the three testbenches with xsim (needs no project); every one must print PASS
#   vivado -mode batch -source scripts/flow.tcl -tclargs program
#       download output/mnist_top.bit to the board over JTAG (board on, USB cable plugged in)
#   vivado -mode batch -source scripts/flow.tcl -tclargs all
#       build, then program the board in the same run (this is what RUN_ME.bat does)
#   vivado -mode batch -source scripts/flow.tcl -tclargs project
#       only create vivado/mnist_cnn.xpr, then open it in the GUI
#
# Run these commands from the project root (the folder that contains rtl/, sim/ ...).
# From the Vivado GUI you can also use  Tools > Run Tcl Script...  (it will build).
# NOTE: the script creates its own project in a short folder (C:/vv_mnist on Windows) and first CLOSES any project that is
#       currently open (your old project is not deleted - reopen its .xpr any time).
# -----------------------------------------------------------------------------

set script_dir [file dirname [file normalize [info script]]]
set root       [file dirname $script_dir]
set rtl_dir    [file join $root rtl]
set proj_name  mnist_cnn
set out_dir    [file join $root output]
set part       xc7a100tcsg324-1

# The Vivado project (runs, checkpoints, temp files) is created in a SHORT folder.  Vivado on Windows fails in
# implementation with confusing errors when a file path gets longer than ~146 characters, which is easy to hit
# in a project unzipped to  Downloads\name\name\...  The sources stay where they are; the bitstream and the
# reports are copied back to output/ . (Set the environment variable MNIST_BUILD_DIR to use another folder.)
if {[info exists ::env(MNIST_BUILD_DIR)]} {
    set proj_dir $::env(MNIST_BUILD_DIR)
} elseif {$::tcl_platform(platform) eq "windows"} {
    set proj_dir C:/vv_mnist
} else {
    set proj_dir /tmp/vv_mnist
}
if {[catch {file mkdir $proj_dir}]} {
    set proj_dir [file join $root vivado]
    file mkdir $proj_dir
}

set mode build
if {[info exists ::argv]} {
    foreach a $::argv {
        if {$a in {project sim build program all}} { set mode $a }
    }
}
puts "== flow.tcl: mode = $mode, root = $root"
puts "==            Vivado project folder = $proj_dir"
if {[string length $root] > 90} {
    puts "== WARNING: the folder of these files has a long path ([string length $root] characters). If anything fails,"
    puts "==          move the whole folder to C:/mnist_cnn and try again."
}
if {[string first " " $root] >= 0} {
    puts "== WARNING: the path of these files contains a space. If anything fails, move the folder to C:/mnist_cnn."
}

# ----------------------------------------------------------------------------- failure reporting
# Prints the first ERROR / CRITICAL WARNING lines of a Vivado log and saves them to output/problem_report.txt
proc first_errors {logfile {max 15}} {
    set found {}
    if {![file exists $logfile]} { return [list "(log file not found: $logfile)"] }
    set fh [open $logfile r]
    set data [read $fh]
    close $fh
    foreach line [split $data "\n"] {
        if {[regexp {^(ERROR|CRITICAL WARNING):} $line]} {
            lappend found [string trim $line]
            if {[llength $found] >= $max} { break }
        }
    }
    if {[llength $found] == 0} {
        lappend found "(no ERROR / CRITICAL WARNING line in $logfile - open it and look at its last 40 lines)"
    }
    return $found
}

proc report_failure {stage logfile} {
    global out_dir root proj_dir
    file mkdir $out_dir
    set rep [file join $out_dir problem_report.txt]
    set lines [first_errors $logfile]
    set fh [open $rep w]
    puts $fh "Failed stage   : $stage"
    puts $fh "Vivado version : [version -short]"
    puts $fh "Files folder   : $root ([string length $root] characters)"
    puts $fh "Project folder : $proj_dir"
    puts $fh "Log file       : $logfile"
    puts $fh ""
    foreach l $lines { puts $fh $l }
    close $fh
    puts "================================================================================"
    puts " [string toupper $stage] FAILED. First messages found in the log:"
    foreach l $lines { puts "   $l" }
    puts " Saved in: $rep   (send me this file, or just the first ERROR line)"
    puts "================================================================================"
    error "$stage failed - see $rep"
}

# ----------------------------------------------------------------------------- project
proc create_proj {} {
    global root rtl_dir proj_dir proj_name part
    catch {close_project}
    file mkdir $proj_dir
    create_project -force $proj_name $proj_dir -part $part
    set_property target_language Verilog [current_project]

    # design sources: the Verilog files AND the .mem data files (weights, deltas, demo images).
    # The .mem files are read by $readmemh during synthesis, so they MUST be part of the project.
    add_files -norecurse [glob -directory $rtl_dir *.v]
    add_files -norecurse [glob -directory $rtl_dir *.mem]
    set_property include_dirs [list $rtl_dir] [get_filesets sources_1]     ;# lets synthesis find the .mem files

    add_files -fileset constrs_1 -norecurse [file join $root constraints nexys_a7_100t.xdc]
    foreach f {mnist_golden tb_cnn_core tb_mnist_top tb_mnist_boot_fail} {
        add_files -fileset sim_1 -norecurse [file join $root sim $f.v]
    }

    set_property top mnist_top    [get_filesets sources_1]
    set_property top tb_mnist_top [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
}

# ----------------------------------------------------------------------------- simulation (xsim, no project needed)
proc run_tool {args} {
    if {[catch {exec -ignorestderr {*}$args} out]} {
        puts $out
        error "command failed: $args"
    }
    return $out
}

proc run_sim {} {
    global root rtl_dir
    set work [file join $root sim_run]
    file delete -force $work
    file mkdir $work
    foreach m [glob -directory $rtl_dir *.mem] { file copy -force $m $work }   ;# $readmemh looks in the run directory
    set ext [expr {$::tcl_platform(platform) eq "windows" ? ".bat" : ""}]
    set srcs [concat [glob -directory $rtl_dir *.v] [list [file join $root sim mnist_golden.v]]]
    cd $work
    foreach tb {tb_cnn_core tb_mnist_boot_fail tb_mnist_top} {
        puts "== xsim: $tb  (this takes a few minutes)"
        run_tool xvlog$ext --relax {*}$srcs [file join $root sim $tb.v]
        run_tool xelab$ext --relax $tb -s ${tb}_snap
        set out [run_tool xsim$ext ${tb}_snap -runall]
        puts $out
        if {![regexp {PASS} $out]} { cd $root; error "$tb did not report PASS" }
    }
    cd $root
    puts "== all testbenches passed"
}

# ----------------------------------------------------------------------------- build
proc run_build {} {
    global root proj_dir proj_name out_dir

    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    set slog [file join $proj_dir $proj_name.runs synth_1 runme.log]
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
        report_failure synthesis $slog
    }
    # a missing .mem file would silently give an all-zero network: fail loudly instead
    set fh [open $slog r]; set txt [read $fh]; close $fh
    if {[regexp {Synth 8-4445|could not open \$readmem} $txt]} {
        puts " A .mem data file could not be read by synthesis - add every file of rtl/ (the .v AND the .mem files) to the project."
        report_failure "synthesis (data file not found)" $slog
    }

    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
        report_failure implementation [file join $proj_dir $proj_name.runs impl_1 runme.log]
    }

    open_run impl_1
    file mkdir $out_dir
    report_utilization    -file [file join $out_dir utilization.rpt]
    report_timing_summary -file [file join $out_dir timing_summary.rpt]
    report_io             -file [file join $out_dir io.rpt]
    file copy -force [file join $proj_dir $proj_name.runs impl_1 mnist_top.bit] [file join $out_dir mnist_top.bit]

    set wns [get_property STATS.WNS [get_runs impl_1]]
    puts "== DONE. bitstream : [file join $out_dir mnist_top.bit]"
    puts "==       reports   : $out_dir  (utilization.rpt, timing_summary.rpt, io.rpt)"
    puts "==       worst setup slack (WNS) = $wns ns"
    if {$wns < 0} {
        puts "== WARNING: timing is NOT met at 100 MHz (WNS < 0). See timing_summary.rpt - send it to me."
    }
}

# ----------------------------------------------------------------------------- program
proc run_program {} {
    global out_dir
    set bit [file join $out_dir mnist_top.bit]
    set marker [file join $out_dir programmed.ok]
    file delete -force $marker
    if {![file exists $bit]} { error "No bitstream at $bit - run the build first" }
    open_hw_manager
    connect_hw_server
    open_hw_target
    set dev [lindex [get_hw_devices xc7a100t*] 0]
    current_hw_device $dev
    set_property PROGRAM.FILE $bit $dev
    program_hw_devices $dev
    puts "== board programmed with $bit"
    close_hw_manager
    set fh [open $marker w]; puts $fh "ok"; close $fh        ;# tells the .bat files that programming worked
}

# ----------------------------------------------------------------------------- dispatch
switch $mode {
    project { create_proj; puts "== project created: [file join $proj_dir $proj_name.xpr]" }
    sim     { run_sim }
    build   { create_proj; run_build }
    program { run_program }
    all     { create_proj; run_build; run_program }
}
