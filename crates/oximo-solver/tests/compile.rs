#[test]
fn typed_translation_boundary() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/ui/fail/*.rs");
}
